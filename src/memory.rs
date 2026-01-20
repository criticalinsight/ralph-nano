use anyhow::{Context, Result, anyhow};
use lancedb::connection::Connection;
use lancedb::table::Table;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use futures::TryStreamExt;
use arrow_array::{RecordBatch, StringArray, FixedSizeListArray, Float32Array, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};

// Candle imports
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{Tokenizer, PaddingParams};
use hf_hub::{api::sync::Api, Repo, RepoType};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphNode {
    pub id: String,
    pub content: String,
    pub node_type: String, // struct, fn, impl, file
    pub path: String,
    pub edges: Vec<String>, // IDs of related nodes
}

pub struct Memory {
    db: Arc<Connection>,
    table: Table,
    // Candle components
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Memory {
    pub async fn new(uri: &str) -> Result<Self> {
        let db = lancedb::connect(uri).execute().await?;

        // Initialize Candle / Metal
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        println!("🧠 Memory Module using device: {:?}", device);

        // Load model from HF Hub (BGE-Small-en-v1.5)
        let model_id = "BAAI/bge-small-en-v1.5".to_string();
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id, RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow!(e))?;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        // Configure tokenizer padding
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // Initialize table if not exists with Graph-over-Vector schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384), false),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false), // JSON: type, path, edges
        ]));

        let table = match db.open_table("memory").execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table("memory", Arc::clone(&schema)).execute().await?
            }
        };

        // Initialize cache table for semantic caching
        let cache_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384), false),
            Field::new("query", DataType::Utf8, false),
            Field::new("response", DataType::Utf8, false),
        ]));
        if db.open_table("cache").execute().await.is_err() {
            let _ = db.create_empty_table("cache", cache_schema).execute().await;
        }

        Ok(Self {
            db: Arc::new(db),
            table,
            model,
            tokenizer,
            device,
        })
    }

    /// Generate embeddings for a batch of texts using Candle
    fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| anyhow!(e))?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        // BGE uses CLS pooling (take the first token, index 0)
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        let embeddings = embeddings.get_on_dim(1, 0)?; // Get index 0 from dim 1 (sequence length)
        
        let embeddings = normalize_l2(&embeddings)?;

        let embeddings_vec: Vec<Vec<f32>> = embeddings.to_vec2()?;
        Ok(embeddings_vec)
    }

    pub async fn create_batch(&self, nodes: Vec<GraphNode>) -> Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = nodes.iter()
            .map(|n| format!("{} {}: {}", n.node_type, n.id, n.content))
            .collect();
        
        let embeddings = self.generate_embeddings(&texts)?;

        let ids = StringArray::from(nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>());
        let contents = StringArray::from(nodes.iter().map(|n| n.content.clone()).collect::<Vec<_>>());
        
        let metadata: Vec<String> = nodes.iter().map(|n| {
            json!({
                "type": n.node_type,
                "path": n.path,
                "edges": n.edges
            }).to_string()
        }).collect();
        let metadata_array = StringArray::from(metadata);

        let vectors = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
            embeddings.iter().map(|v| Some(v.iter().map(|&x| Some(x)))),
            384
        );

        let schema = self.table.schema().await?;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(vectors),
                Arc::new(contents),
                Arc::new(metadata_array),
            ],
        )?;

        self.table.add(Box::new(RecordBatchIterator::new(vec![Ok(batch)], self.table.schema().await?))).execute().await?;

        Ok(())
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<GraphNode>> {
        let embeddings = self.generate_embeddings(&[query.to_string()])?;
        let query_vector = embeddings.first().ok_or(anyhow!("Failed to embed query"))?;

        let results = self.table
            .query()
            .nearest_to(query_vector.clone())?
            .limit(limit)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut nodes = Vec::new();
        for batch in results {
            let ids: &StringArray = batch.column_by_name("id").unwrap().as_any().downcast_ref().unwrap();
            let contents: &StringArray = batch.column_by_name("content").unwrap().as_any().downcast_ref().unwrap();
            let metadatas: &StringArray = batch.column_by_name("metadata").unwrap().as_any().downcast_ref().unwrap();

            for i in 0..batch.num_rows() {
                let id = ids.value(i).to_string();
                let content = contents.value(i).to_string();
                let meta_str = metadatas.value(i);
                let meta: Value = serde_json::from_str(meta_str)?;

                nodes.push(GraphNode {
                    id,
                    content,
                    node_type: meta["type"].as_str().unwrap_or("").to_string(),
                    path: meta["path"].as_str().unwrap_or("").to_string(),
                    edges: meta["edges"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_str().unwrap().to_string()).collect(),
                });
            }
        }

        Ok(nodes)
    }

    pub async fn store_fact(&self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        let content = format!("{} {} {}", subject, predicate, object);
        let id = uuid::Uuid::new_v4().to_string();
        
        let node = GraphNode {
            id,
            content,
            node_type: "fact".to_string(),
            path: "global/memory".to_string(),
            edges: vec![],
        };

        self.create_batch(vec![node]).await
    }

    pub async fn add_node(&self, node: GraphNode) -> Result<()> {
        self.create_batch(vec![node]).await
    }

    pub async fn store_lesson(&self, triple: &str) -> Result<()> {
        // Simple storage of the triple as a lesson
        let node = GraphNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: triple.to_string(),
            node_type: "lesson".to_string(),
            path: "global/memory".to_string(),
            edges: vec![],
        };
        self.create_batch(vec![node]).await
    }

    pub async fn store_heuristic(&self, content: &str) -> Result<()> {
        let node = GraphNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            node_type: "heuristic".to_string(),
            path: "global/memory".to_string(),
            edges: vec![],
        };
        self.create_batch(vec![node]).await
    }

    pub async fn recall_heuristics(&self, query: &str) -> Result<Vec<String>> {
        let nodes = self.search(query, 5).await?;
        let heuristics: Vec<String> = nodes.into_iter()
            .filter(|n| n.node_type == "heuristic" || n.node_type == "lesson")
            .map(|n| n.content)
            .collect();
        Ok(heuristics)
    }

    pub async fn recall_facts(&self, query: &str) -> Result<Vec<String>> {
        let nodes = self.search(query, 5).await?;
        let facts: Vec<String> = nodes.into_iter()
            .filter(|n| n.node_type == "fact")
            .map(|n| n.content)
            .collect();
        Ok(facts)
    }

    /// Check cache for a semantically similar query
    pub async fn check_cache(&self, query: &str) -> Result<Option<String>> {
        let cache_table = match self.db.open_table("cache").execute().await {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        let embeddings = self.generate_embeddings(&[query.to_string()])?;
        let query_vector = embeddings.first().ok_or(anyhow!("Failed to embed query"))?;

        let results = cache_table
            .query()
            .nearest_to(query_vector.clone())?
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        for batch in results {
            if batch.num_rows() == 0 {
                return Ok(None);
            }
            // Check distance. LanceDB returns _distance column.
            if let Some(dist_col) = batch.column_by_name("_distance") {
                if let Some(dist_arr) = dist_col.as_any().downcast_ref::<Float32Array>() {
                    let distance = dist_arr.value(0);
                    if distance < 0.1 { // Strict threshold for cache hit
                        let responses: &StringArray = batch.column_by_name("response").unwrap().as_any().downcast_ref().unwrap();
                        return Ok(Some(responses.value(0).to_string()));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Store a query-response pair in the cache
    pub async fn store_cache(&self, query: &str, response: &str) -> Result<()> {
        let cache_table = self.db.open_table("cache").execute().await?;

        let embeddings = self.generate_embeddings(&[query.to_string()])?;
        let id = uuid::Uuid::new_v4().to_string();

        let ids = StringArray::from(vec![id]);
        let queries = StringArray::from(vec![query.to_string()]);
        let responses = StringArray::from(vec![response.to_string()]);
        
        let vectors = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
            embeddings.iter().map(|v| Some(v.iter().map(|&x| Some(x)))),
            384
        );

        let schema = cache_table.schema().await?;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(ids),
                Arc::new(vectors),
                Arc::new(queries),
                Arc::new(responses),
            ],
        )?;

        cache_table.add(Box::new(RecordBatchIterator::new(vec![Ok(batch)], cache_table.schema().await?))).execute().await?;
        Ok(())
    }
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let sum_sq = v.sqr()?.sum_keepdim(1)?;
    let norm = sum_sq.sqrt()?;
    Ok(v.broadcast_div(&norm)?)
}
