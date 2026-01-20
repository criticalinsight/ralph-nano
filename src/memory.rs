use anyhow::{Context, Result, anyhow};
use cozo::{DbInstance, ScriptMutability, DataValue, Vector};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

// Candle imports
use candle_core::{Device, Tensor, DType};
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;
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
    db: DbInstance,
    // Candle components
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Memory {
    pub async fn new(path: &str) -> Result<Self> {
        let db_path = Path::new(path).join("cozo.db");
        let db = DbInstance::new("rocksdb", db_path.to_str().unwrap(), Default::default())
            .map_err(|e| anyhow!("Failed to open CozoDB: {}", e))?;

        // Initialize Schema
        let create_cache = "
            :create cache {
                id: String
                =>
                query: String,
                response: String,
                embedding: <F32; 384>
            }
        ";
        if let Err(e) = db.run_script(create_cache, Default::default(), ScriptMutability::Mutable) {
            println!("⚠️ Cache table creation warning: {}", e);
        }

        let create_index = "
            ::hnsw create cache:idx {
                dim: 384,
                dtype: F32,
                fields: [embedding],
                distance: Cosine,
                m: 50,
                ef_construction: 200
            }
        ";
        let _ = db.run_script(create_index, Default::default(), ScriptMutability::Mutable);

        let create_nodes = "
            :create nodes {
                id: String
                =>
                content: String,
                type: String,
                path: String
            }
        ";
        let _ = db.run_script(create_nodes, Default::default(), ScriptMutability::Mutable);

        let create_edges = "
            :create edges {
                from: String,
                to: String
                =>
                rel_type: String
            }
        ";
        let _ = db.run_script(create_edges, Default::default(), ScriptMutability::Mutable);

        // Initialize Library Table for Docs (Enhanced)
        let create_library = "
            :create library {
                id: String
                =>
                name: String,
                version: String,
                content: String,
                language: String,
                chunk_type: String,
                embedding: <F32; 384>
            }
        ";
        if let Err(e) = db.run_script(create_library, Default::default(), ScriptMutability::Mutable) {
            println!("⚠️ Library table creation warning: {}", e);
        }

        let create_kv = "
            :create kv_cache {
                hash: String
                =>
                cache_id: String,
                created_at: Int
            }
        ";
        let _ = db.run_script(create_kv, Default::default(), ScriptMutability::Mutable);

        let create_sync = "
            :create sync_log {
                path: String
                =>
                last_ingested: Int,
                content_hash: String
            }
        ";
        let _ = db.run_script(create_sync, Default::default(), ScriptMutability::Mutable);

        let create_lib_index = "
            ::hnsw create library:idx {
                dim: 384,
                dtype: F32,
                fields: [embedding],
                distance: Cosine,
                m: 50,
                ef_construction: 200
            }
        ";
        let _ = db.run_script(create_lib_index, Default::default(), ScriptMutability::Mutable);

        // Initialize Candle / CPU (Metal has missing kernels for layer-norm in some versions)
        let device = Device::Cpu;
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
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        }

        Ok(Self {
            db,
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true).map_err(|e| anyhow!(e))?;
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)?.unsqueeze(0)?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        let cls_embedding = embeddings.get_on_dim(1, 0)?;
        let vector: Vec<f32> = cls_embedding.flatten_all()?.to_vec1()?;
        
        Ok(normalize_l2(&vector))
    }

    pub fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }
        
        // Encode all texts
        let mut results = Vec::new();
        for text in texts {
            results.push(self.embed(text)?);
        }
        Ok(results)
    }

    pub async fn check_cache(&self, query: &str) -> Result<Option<String>> {
        let embedding = self.embed(query)?;
        
        let query_script = "
            ?[response] := ~cache:idx {
                response |
                query: $query_vec,
                k: 1,
                bind_distance: dist,
                ef: 100
            },
            dist < 0.1
        ";
        
        let mut params = BTreeMap::new();
        params.insert("query_vec".to_string(), vec_to_datavalue(embedding));

        let result = self.db.run_script(query_script, params, ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Cache query failed: {}", e))?;

        if let Some(row) = result.rows.first() {
            if let Some(val) = row.first() {
                if let DataValue::Str(s) = val {
                    return Ok(Some(s.to_string()));
                }
            }
        }

        Ok(None)
    }

    pub async fn store_cache(&self, query: &str, response: &str) -> Result<()> {
        let embedding = self.embed(query)?;
        let id = uuid::Uuid::new_v4().to_string();
        
        let query_script = "
            ?[id, query, response, embedding] <- [[$id, $query, $response, $embedding]]
            :put cache { id => query, response, embedding }
        ";
       
        let mut params = BTreeMap::new();
        params.insert("id".to_string(), DataValue::from(id));
        params.insert("query".to_string(), DataValue::from(query));
        params.insert("response".to_string(), DataValue::from(response));
        params.insert("embedding".to_string(), vec_to_datavalue(embedding));

        self.db.run_script(query_script, params, ScriptMutability::Mutable)
            .map_err(|e| anyhow!("Failed to store cache: {}", e))?;
        
        Ok(())
    }

    pub async fn add_node(&self, node: &GraphNode) -> Result<()> {
        let query_script = "
            ?[id, content, type, path] <- [[$id, $content, $type, $path]]
            :put nodes { id => content, type, path }
        ";
       
        let mut params = BTreeMap::new();
        params.insert("id".to_string(), DataValue::from(node.id.clone()));
        params.insert("content".to_string(), DataValue::from(node.content.clone()));
        params.insert("type".to_string(), DataValue::from(node.node_type.clone()));
        params.insert("path".to_string(), DataValue::from(node.path.clone()));

        self.db.run_script(query_script, params, ScriptMutability::Mutable)
            .map_err(|e| anyhow!("Failed to add node: {}", e))?;
            
        // Also store edges
        if !node.edges.is_empty() {
            let edge_script = "
                ?[from, to, rel_type] <- [[$from, $to, $rel]]
                :put edges { from, to => rel_type }
            ";
            for target in &node.edges {
                let mut edge_params = BTreeMap::new();
                edge_params.insert("from".to_string(), DataValue::from(node.id.clone()));
                edge_params.insert("to".to_string(), DataValue::from(target.clone()));
                edge_params.insert("rel".to_string(), DataValue::from("related"));
                
                self.db.run_script(edge_script, edge_params, ScriptMutability::Mutable)
                     .map_err(|e| anyhow!("Failed to store edge: {}", e))?;
            }
        }
        
        Ok(())
    }

    pub async fn store_lesson(&self, lesson: &str) -> Result<()> {
        let id = uuid::Uuid::new_v4().to_string();
        self.add_node(&GraphNode {
            id,
            content: lesson.to_string(),
            node_type: "lesson".to_string(),
            path: "global".to_string(),
            edges: vec![],
        }).await
    }

    pub async fn store_heuristic(&self, heuristic: &str) -> Result<()> {
         let id = uuid::Uuid::new_v4().to_string();
         self.add_node(&GraphNode {
            id,
            content: heuristic.to_string(),
            node_type: "heuristic".to_string(),
            path: "global".to_string(),
            edges: vec![],
        }).await
    }
    
    pub async fn recall_facts(&self, _query: &str) -> Result<Vec<String>> {
        // Placeholder: For now return all lessons. Ideally would be semantic search.
        let script = "?[content] := *nodes{content, type}, type = 'lesson'";
        let result = self.db.run_script(script, Default::default(), ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Failed to recall facts: {}", e))?;
            
        let mut facts = Vec::new();
        for row in result.rows {
            if let Some(DataValue::Str(s)) = row.first() {
                facts.push(s.to_string());
            }
        }
        Ok(facts)
    }

    pub async fn recall_heuristics(&self, _query: &str) -> Result<Vec<String>> {
        let script = "?[content] := *nodes{content, type}, type = 'heuristic'";
        let result = self.db.run_script(script, Default::default(), ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Failed to recall heuristics: {}", e))?;
            
        let mut heuristics = Vec::new();
        for row in result.rows {
            if let Some(DataValue::Str(s)) = row.first() {
                heuristics.push(s.to_string());
            }
        }
        Ok(heuristics)
    }

    pub async fn add_library_entry(&self, id: &str, name: &str, version: &str, content: &str, lang: &str, chunk_type: &str) -> Result<()> {
        let embedding = self.embed(content)?;
        self.batch_add_library_entries(vec![(id.to_string(), name.to_string(), version.to_string(), content.to_string(), lang.to_string(), chunk_type.to_string(), embedding)]).await
    }

    pub async fn batch_add_library_entries(&self, entries: Vec<(String, String, String, String, String, String, Vec<f32>)>) -> Result<()> {
        if entries.is_empty() { return Ok(()); }
        
        let query_script = "
            ?[id, name, version, content, language, chunk_type, embedding] <- $data
            :put library { id => name, version, content, language, chunk_type, embedding }
        ";
       
        let mut data_rows = Vec::new();
        for (id, name, version, content, lang, chunk_type, embedding) in entries {
            let row = vec![
                DataValue::from(id),
                DataValue::from(name),
                DataValue::from(version),
                DataValue::from(content),
                DataValue::from(lang),
                DataValue::from(chunk_type),
                vec_to_datavalue(embedding)
            ];
            data_rows.push(DataValue::List(row));
        }

        let mut params = BTreeMap::new();
        params.insert("data".to_string(), DataValue::List(data_rows));

        self.db.run_script(query_script, params, ScriptMutability::Mutable)
            .map_err(|e| anyhow!("Failed to bulk add library entries: {}", e))?;
        
        Ok(())
    }

    pub async fn search_library(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let embedding = self.embed(query)?;
        
        // Weighted Search: Prioritize 'definition' and 'module' types
        let query_script = format!("
            ?[content, type, dist] := ~library:idx {{
                content, chunk_type: type |
                query: $query_vec,
                k: {},
                bind_distance: dist,
                ef: 100
            }}
            
            // Apply weight: boost definitions by reducing their perceived distance
            ?[content, score] := ?[content, type, dist],
                weight = if type == 'definition' {{ 0.8 }} else {{ 1.0 }},
                score = dist * weight
            
            :sort score
            :limit {}
        ", limit * 2, limit);
        
        let mut params = BTreeMap::new();
        params.insert("query_vec".to_string(), vec_to_datavalue(embedding));

        let result = self.db.run_script(&query_script, params, ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Library search failed: {}", e))?;

        let mut results = Vec::new();
        for row in result.rows {
            if let Some(DataValue::Str(s)) = row.first() {
                results.push(s.to_string());
            }
        }

        Ok(results)
    }

    pub async fn get_known_libraries(&self) -> Result<Vec<String>> {
        let script = "?[name] := *library{name}";
        let result = self.db.run_script(script, Default::default(), ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Failed to get known libraries: {}", e))?;
            
        let mut names = Vec::new();
        for row in result.rows {
            if let Some(DataValue::Str(s)) = row.first() {
                names.push(s.to_string());
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    pub async fn get_kv_cache(&self, hash: &str) -> Result<Option<String>> {
        let script = "?[cache_id] := kv_cache { hash: $hash, cache_id }";
        let mut params = BTreeMap::new();
        params.insert("hash".to_string(), DataValue::from(hash.to_string()));

        let res = self.db.run_script(script, params, ScriptMutability::Immutable)
            .map_err(|e| anyhow!("KV Lookup failed: {}", e))?;
        
        if let Some(row) = res.rows.first() {
            if let Some(DataValue::Str(s)) = row.first() {
                return Ok(Some(s.to_string()));
            }
        }
        Ok(None)
    }

    pub async fn set_kv_cache(&self, hash: &str, cache_id: &str) -> Result<()> {
        let script = "
            ?[hash, cache_id, created_at] <- [[$hash, $cache_id, $now]]
            :put kv_cache { hash => cache_id, created_at }
        ";
        let now = chrono::Utc::now().timestamp() as i64;
        let mut params = BTreeMap::new();
        params.insert("hash".to_string(), DataValue::from(hash.to_string()));
        params.insert("cache_id".to_string(), DataValue::from(cache_id.to_string()));
        params.insert("now".to_string(), DataValue::from(now));

        self.db.run_script(script, params, ScriptMutability::Mutable)
            .map_err(|e| anyhow!("KV Store failed: {}", e))?;
        Ok(())
    }

    pub async fn get_neighborhood(&self, path: &str) -> Result<Vec<String>> {
        // Query nodes in the same file + direct neighbors linked via edges
        let script = "
            ?[content] := *nodes{path, content}, path = $path
            ?[content] := *nodes{id, content}, *edges{from: $path, to: id}
            ?[content] := *nodes{id, content}, *edges{from: id, to: $path}
        ";
        let mut params = BTreeMap::new();
        params.insert("path".to_string(), DataValue::from(path.to_string()));

        let res = self.db.run_script(script, params, ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Neighborhood query failed: {}", e))?;
        
        let mut results = Vec::new();
        for row in res.rows {
            if let Some(DataValue::Str(s)) = row.first() {
                results.push(s.to_string());
            }
        }
        Ok(results)
    }

    pub async fn check_sync_status(&self, path: &str) -> Result<Option<(i64, String)>> {
        let script = "?[last_ingested, content_hash] := sync_log { path: $path, last_ingested, content_hash }";
        let mut params = BTreeMap::new();
        params.insert("path".to_string(), DataValue::from(path.to_string()));

        let res = self.db.run_script(script, params, ScriptMutability::Immutable)
            .map_err(|e| anyhow!("Sync lookup failed: {}", e))?;
        
        if let Some(row) = res.rows.first() {
            // Extract timestamp - Cozo stores numbers as Num which wraps i64/f64
            let ts = match row.get(0) {
                Some(DataValue::Num(n)) => {
                    // Num can be Int or Float - use Debug string parsing as fallback
                    let s = format!("{:?}", n);
                    s.parse::<i64>().unwrap_or(0)
                },
                _ => 0,
            };
            let hash = match row.get(1) {
                Some(DataValue::Str(h)) => h.to_string(),
                _ => String::new(),
            };
            if !hash.is_empty() {
                return Ok(Some((ts, hash)));
            }
        }
        Ok(None)
    }

    pub async fn update_sync_status(&self, path: &str, hash: &str) -> Result<()> {
        let script = "
            ?[path, last_ingested, content_hash] <- [[$path, $now, $hash]]
            :put sync_log { path => last_ingested, content_hash }
        ";
        let now = chrono::Utc::now().timestamp() as i64;
        let mut params = BTreeMap::new();
        params.insert("path".to_string(), DataValue::from(path.to_string()));
        params.insert("hash".to_string(), DataValue::from(hash.to_string()));
        params.insert("now".to_string(), DataValue::from(now));

        self.db.run_script(script, params, ScriptMutability::Mutable)
            .map_err(|e| anyhow!("Sync update failed: {}", e))?;
        Ok(())
    }

    /// Multi-turn Re-ranking for High Precision (Phase 5/6)
    pub async fn rerank(&self, query: &str, candidates: Vec<String>, limit: usize) -> Result<Vec<String>> {
        if candidates.is_empty() { return Ok(vec![]); }
        
        // Local cross-encoding simulation (using BGE embeddings dot-product for now, 
        // ideally would be a real cross-encoder model like BGE-Reranker)
        let query_vec = self.embed(query)?;
        let mut scored_candidates = Vec::new();

        for content in candidates {
            let content_vec = self.embed(&content)?;
            let dot_product: f32 = query_vec.iter().zip(content_vec.iter()).map(|(a, b)| a * b).sum();
            scored_candidates.push((content, dot_product));
        }

        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored_candidates.into_iter().take(limit).map(|(c, _)| c).collect())
    }
}

fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-6 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn vec_to_datavalue(v: Vec<f32>) -> DataValue {
    DataValue::Vec(Vector::F32(Array1::from(v)))
}

// Added unit tests helper
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_semantic_cache() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("ralph_test_cozo");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir)?;
        }
        std::fs::create_dir_all(&temp_dir)?;
        
        // Use a persistent path for rocksdb
        let memory = Memory::new(temp_dir.to_str().unwrap()).await?;

        // Test Miss
        assert!(memory.check_cache("Capital of France?").await?.is_none());

        // Test Store
        memory.store_cache("Capital of France?", "Paris").await?;

        // Test Hit
        let res = memory.check_cache("Capital of France?").await?;
        assert_eq!(res, Some("Paris".to_string()));
        
        Ok(())
    }
}
