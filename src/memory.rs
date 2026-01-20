use anyhow::{Context, Result};
use lancedb::connection::Connection;
use lancedb::table::Table;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use futures::TryStreamExt;
use arrow_array::{RecordBatch, StringArray, FixedSizeListArray, Float32Array, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};

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
    embedder: TextEmbedding,
}

impl Memory {
    pub async fn new(uri: &str) -> Result<Self> {
        let db = lancedb::connect(uri).execute().await?;
        let mut options = InitOptions::default();
        options.model_name = EmbeddingModel::BGESmallENV15;
        let embedder = TextEmbedding::try_new(options)?;

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

        Ok(Self {
            db: Arc::new(db),
            table,
            embedder,
        })
    }

    pub async fn add_node(&self, node: GraphNode) -> Result<()> {
        let vectors = self.embedder.embed(vec![&node.content], None)?;
        let vector = vectors[0].clone();
        
        let metadata = json!({
            "type": node.node_type,
            "path": node.path,
            "edges": node.edges,
        });

        let batch = self.create_batch(&node.id, &vector, &node.content, &metadata.to_string())?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.table.add(reader).execute().await?;
        
        Ok(())
    }

    pub async fn store_lesson(&self, triple: &str) -> Result<()> {
        // Optimization: Prepend "FACT:" to help the embedding model distinguish
        let fact_content = format!("FACT: {}", triple);
        let vectors = self.embedder.embed(vec![&fact_content], None)?;
        let vector = vectors[0].clone();

        let metadata = json!({
            "type": "fact",
            "path": "janitor",
            "edges": Vec::<String>::new(),
        });

        let batch = self.create_batch(triple, &vector, triple, &metadata.to_string())?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.table.add(reader).execute().await?;
        Ok(())
    }

    pub async fn store_heuristic(&self, heuristic: &str) -> Result<()> {
        let content = format!("HEURISTIC: {}", heuristic);
        let vectors = self.embedder.embed(vec![&content], None)?;
        let vector = vectors[0].clone();

        let metadata = json!({
            "type": "heuristic",
            "path": "reflexion",
            "edges": Vec::<String>::new(),
        });

        let batch = self.create_batch(heuristic, &vector, heuristic, &metadata.to_string())?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.table.add(reader).execute().await?;
        Ok(())
    }

    pub async fn recall_heuristics(&self, query: &str) -> Result<Vec<String>> {
        let heuristic_query = format!("HEURISTIC: {}", query);
        let vectors = self.embedder.embed(vec![&heuristic_query], None)?;
        let query_vec = vectors[0].clone();

        let mut stream = self.table
            .query()
            .nearest_to(query_vec.as_slice())?
            .limit(5)
            .execute()
            .await?;

        let mut heuristics = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let content_col = batch.column(2).as_any().downcast_ref::<StringArray>().context("content col")?;
            let metadata_col = batch.column(3).as_any().downcast_ref::<StringArray>().context("meta col")?;

            for i in 0..batch.num_rows() {
                let metadata: Value = serde_json::from_str(metadata_col.value(i))?;
                if metadata["type"] == "heuristic" {
                    heuristics.push(content_col.value(i).to_string());
                }
            }
        }
        Ok(heuristics)
    }

    fn create_batch(&self, id: &str, vector: &[f32], content: &str, metadata: &str) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384), false),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false),
        ]));

        let id_arr = Arc::new(StringArray::from(vec![id])) as Arc<dyn arrow_array::Array>;
        let content_arr = Arc::new(StringArray::from(vec![content])) as Arc<dyn arrow_array::Array>;
        let metadata_arr = Arc::new(StringArray::from(vec![metadata])) as Arc<dyn arrow_array::Array>;
        
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Arc::new(Float32Array::from(vector.to_vec())) as Arc<dyn arrow_array::Array>;
        let vector_arr = Arc::new(FixedSizeListArray::new(
            field,
            384,
            values,
            None
        )) as Arc<dyn arrow_array::Array>;

        RecordBatch::try_new(schema, vec![
            id_arr,
            vector_arr,
            content_arr,
            metadata_arr,
        ]).context("Failed to create record batch")
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<GraphNode>> {
        let vectors = self.embedder.embed(vec![query], None)?;
        let query_vec = vectors[0].clone();
        
        let mut stream = self.table
            .query()
            .nearest_to(query_vec.as_slice())?
            .limit(limit)
            .execute()
            .await?;

        let mut nodes = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let id_col = batch.column(0).as_any().downcast_ref::<StringArray>().context("id col")?;
            let content_col = batch.column(2).as_any().downcast_ref::<StringArray>().context("content col")?;
            let metadata_col = batch.column(3).as_any().downcast_ref::<StringArray>().context("meta col")?;

            for i in 0..batch.num_rows() {
                let metadata: Value = serde_json::from_str(metadata_col.value(i))?;
                nodes.push(GraphNode {
                    id: id_col.value(i).to_string(),
                    content: content_col.value(i).to_string(),
                    node_type: metadata["type"].as_str().unwrap_or("unknown").to_string(),
                    path: metadata["path"].as_str().unwrap_or("unknown").to_string(),
                    edges: metadata["edges"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect()).unwrap_or_default(),
                });
            }
        }
        Ok(nodes)
    }

    pub async fn recall_facts(&self, query: &str) -> Result<Vec<String>> {
        let fact_query = format!("FACT: {}", query);
        let vectors = self.embedder.embed(vec![&fact_query], None)?;
        let query_vec = vectors[0].clone();

        let mut stream = self.table
            .query()
            .nearest_to(query_vec.as_slice())?
            .limit(5)
            .execute()
            .await?;

        let mut facts = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            let content_col = batch.column(2).as_any().downcast_ref::<StringArray>().context("content col")?;
            let metadata_col = batch.column(3).as_any().downcast_ref::<StringArray>().context("meta col")?;

            for i in 0..batch.num_rows() {
                let metadata: Value = serde_json::from_str(metadata_col.value(i))?;
                if metadata["type"] == "fact" {
                    facts.push(content_col.value(i).to_string());
                }
            }
        }
        Ok(facts)
    }

    pub async fn get_neighbors(&self, _node_id: &str) -> Result<Vec<GraphNode>> {
        // Search by ID filter to get edges, then search those IDs
        Ok(vec![])
    }
}
