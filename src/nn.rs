#[path = "./function.rs"]
mod function;
use byteorder::{LittleEndian, ReadBytesExt};
use function::*;
#[path = "./tokenizer.rs"]
mod tokenizer;
use log::info;
use tokenizer::{Token, Tokenizer};
use std::{fs::File, io::Read};
use std::cmp::max;
use rand::prelude::*;
use rand::distributions::WeightedIndex;

struct Sampler {
    embedding_t: Matrix,
    final_logit_softcapping: f32

}

impl Sampler {
    pub fn new(embedding:&Embedding) -> Sampler {
        let mut embedding_t = embedding.weight.clone();
        embedding_t.transpose();
        Sampler {
            embedding_t,
            final_logit_softcapping: 30.0
        }
    }

    fn _weighted_random_select(normed_logits: &Vec<(u32, f32)>) -> Option<usize> {
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(normed_logits.iter().map(|token| token.1)).ok()?;
        Some(dist.sample(&mut rng))
    }

    pub fn forward(
        &self,
        hidden_state: &Matrix,
        temperature: f32,
        top_p: f32,
        top_k: usize
    ) -> u32 {
        assert!(temperature >= 0.0, "Temperature can not be negtive.");
        let mut logits = matmul(&hidden_state, &self.embedding_t); // [1, vocab_size]
        for i in logits.data.iter_mut() {
            *i /= self.final_logit_softcapping;
            *i = i.tanh();
            *i *= self.final_logit_softcapping;
            *i /= temperature;
        }
        logits.softmax();
        let mut tokens = vec![];
        for i in 0..logits.data.len() {
            tokens.push((i as u32, logits.data[i]));
        }
        tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // apply top_p
        let mut cumsum_p = 0.0f32;
        let mut cnt = 0;
        for i in tokens.iter() {
            if cumsum_p <= top_p {
                cumsum_p += i.1;
                cnt += 1;
            } else {
                break;
            }
        }
        // apply top_k
        cnt = max(cnt, top_k);

        // re-norm
        let filtered_tokens = tokens[0..cnt].to_vec();
        let sum_prob = filtered_tokens.iter().fold(0.0, |prob, &x| prob+x.1);
        let normed_tokens: Vec<(u32, f32)> = filtered_tokens
            .iter()
            .map(|&(index, prob)| (index, prob / sum_prob))
            .collect();

        // weighted select
        let selected_index:usize = Self::_weighted_random_select(&normed_tokens).unwrap();

        let output_token_index:u32 = normed_tokens[selected_index].0;
        output_token_index
    }
}




struct Linear {
    weight_t: Matrix,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(weight_data: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert!(weight_data.len() == in_features*out_features,
            "Size of weight data imcompatable for Linear");
        // weight_t is the transpose of weight_data
        let mut weight_t = Matrix::new(weight_data, out_features, in_features);
        weight_t.transpose();
        Linear {
            weight_t,
            in_features,
            out_features
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        matmul(input, &self.weight_t)
    }
}


struct GemmaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear
}

impl GemmaMLP {
    pub fn new(weight_data: Vec<f32>, hidden_size:usize, intermediate_size:usize) -> Self {
        let len = hidden_size*intermediate_size;
        assert!(weight_data.len() == len*3,
            "Size of weight data imcompatable for GemmaMLP");

        GemmaMLP {
            down_proj : Linear::new(weight_data[..len].to_vec(), intermediate_size, hidden_size),
            gate_proj : Linear::new(weight_data[len..2*len].to_vec(), hidden_size, intermediate_size),
            up_proj : Linear::new(weight_data[2*len..].to_vec(), hidden_size, intermediate_size)
        }
    }

    pub fn forward(&self, x:&Matrix) -> Matrix {
        let mut gate = self.gate_proj.forward(x);
        gelu(&mut gate); // F.gelu(gate, approximate="tanh")
        let up = self.up_proj.forward(x);
        let fuse = matmul(&gate, &up);
        let outputs = self.down_proj.forward(&fuse);
        outputs
    }

}


struct Embedding {
    weight: Matrix,
    num_embeddings: usize,
    embedding_dim: usize
}

impl Embedding {
    pub fn new(weight_data:Vec<f32>, num_embeddings: usize, embedding_dim: usize) -> Embedding {
        let weight = Matrix::new(weight_data, num_embeddings, embedding_dim);
        Embedding {
            weight,
            num_embeddings,
            embedding_dim
        }

    }

    pub fn forward(self, input_ids:&Vec<u32>) -> Matrix {
        let mut output = vec![];
        for i in 0..input_ids.len() {
            output.push(self.weight.data[i]);
        }
        Matrix::new(output, input_ids.len(), self.embedding_dim)
    }

    pub fn from_fp(fp:&mut File) -> Embedding {
        let num_embeddings = 256000;
        let embedding_dim = 2304;
        let weight_data = read_next_n_fp32(fp, num_embeddings*embedding_dim).unwrap(); // vocab_size*hidden_size
        Embedding::new(weight_data, num_embeddings, embedding_dim)
    }
}


struct RMSNorm {
    weight: Matrix,
    dim: usize,
    eps: f32
}

impl RMSNorm {
    pub fn new(weight_data:Vec<f32>, dim:usize) -> RMSNorm {
        let mut weight = Matrix::new(weight_data, dim, 1);
        for i in 0..weight.data.len() { weight.data[i] += 1.0; }
        RMSNorm {
            weight,
            dim,
            eps: 1e-6
        }
    }
    fn _norm(&self, x:&mut Matrix) {
        // x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        for r in 0..x.n_row {
            let mut sq_mean:f32 = 0.0;
            for c in 0..x.n_col {
                sq_mean += x.get(r, c).powi(2);
            }
            sq_mean /= x.n_col as f32;
            sq_mean += self.eps;
            sq_mean = sq_mean.sqrt();
            for c in 0..x.n_col {
                x.data[r*x.n_col+c] /= sq_mean;
            }
        }
    }
    pub fn forward(&self, x:&Matrix) -> Matrix {
        let x = &mut x.clone();
        self._norm(x);
        let output = matmul(x, &self.weight);
        assert!(output.n_row == x.n_row && output.n_col == x.n_col);
        output
    }
}


struct GemmaAttention {
    num_heads:usize,
    num_kv_heads:usize,
    num_queries_per_kv:usize,
    hidden_size:usize,
    head_dim:usize,
    q_size:usize,
    kv_size:usize,
    scaling:f32,
    q_proj:Linear,
    k_proj:Linear,
    v_proj:Linear,
    o_proj:Linear,
    attn_logit_softcapping:f32
}

impl GemmaAttention {
    pub fn new(
        weight_data:Vec<f32>,
        num_heads:usize,
        num_kv_heads:usize,
        head_dim:usize,
        query_pre_attn_scalar:usize,
        hidden_size:usize,
        attn_logit_softcapping:f32
    ) -> GemmaAttention {
        assert!(num_heads % num_kv_heads == 0);
        let num_queries_per_kv = num_heads / num_kv_heads;

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let scaling:f32 = 1.0 / (query_pre_attn_scalar as f32).sqrt();

        let mut cursor:usize = 0;
        let q_weight = weight_data[0..hidden_size*q_size].to_vec();
        cursor += q_weight.len();
        let k_weight = weight_data[cursor..cursor+hidden_size*kv_size].to_vec();
        cursor += k_weight.len();
        let v_weight = weight_data[cursor..cursor+hidden_size*kv_size].to_vec();
        cursor += v_weight.len();
        let o_weight = weight_data[cursor..cursor+hidden_size*q_size].to_vec();
        cursor += o_weight.len();
        assert!(cursor == weight_data.len());

        let q_proj = Linear::new(q_weight, hidden_size, q_size);
        let k_proj = Linear::new(k_weight, hidden_size, kv_size);
        let v_proj = Linear::new(v_weight, hidden_size, kv_size);
        let o_proj = Linear::new(o_weight, q_size, hidden_size);

        GemmaAttention {
            num_heads,
            num_kv_heads,
            num_queries_per_kv,           
            hidden_size,
            head_dim,
            q_size,
            kv_size,
            scaling,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attn_logit_softcapping
        }
    }

    fn _chunked_xq_by_heads(xq:Matrix, head_dim:usize) -> Vec<Matrix> {
        assert!(xq.n_row == 1 && xq.n_col % head_dim == 0);
        let mut chunked_q:Vec<Matrix> = vec![];
        for i in 0..xq.data.len() {
            chunked_q.push(
                Matrix::new(xq.data[i*head_dim..(i+1)*head_dim].to_vec(), 1, head_dim)
            );
        }
        chunked_q
    }
    fn _repeat_xk(xk:&mut Matrix, times:usize) {
        assert!(xk.n_row == 1);
        // ori_k [1, kv_size]
        for _ in 1..times {
            let _k = Matrix::new(xk.data.clone(), 1, xk.n_col);
            xk.concat(_k, 0);
        }
    }
    fn _apply_rope(x: &mut Matrix, pos: usize, num_heads:usize, head_dim: usize) {
        assert!(x.n_row == 1 && x.n_col == num_heads*head_dim);

        for i in 0..num_heads {
            let dim_base_offset = i*head_dim;

            for j in 0..head_dim/2 {
                let theta = (pos as f32) / (10000.0_f32.powf(2.0 * j as f32 / head_dim as f32));
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = x.data[2*j+dim_base_offset];
                let x1 = x.data[2*j+1+dim_base_offset];

                x.data[2*j+dim_base_offset] = cos_theta*x0 - sin_theta*x1;
                x.data[2*j+1+dim_base_offset] = sin_theta*x0 + cos_theta*x1;
            }
        }
    }
    fn _add_to_k_cache(&self, k_cache: &mut Vec<Matrix>, xk:Matrix) {
        // k_cache: [head_dim, seq_len] * num_kv_heads
        // xk: [1, kv_size] kv_size = num_kv_heads * head_dim
        for i in 0..self.num_kv_heads {
            let range = i*self.head_dim..(i+1)*self.head_dim;
            let head_xk = Matrix::new(xk.data[range].to_vec(), self.head_dim, 1);
            k_cache[i].concat(head_xk, 1);
        }
    }
    fn _add_to_v_cache(&self, v_cache: &mut Vec<Matrix>, xv:Matrix) {
        // v_cache: [seq_len, head_dim] * num_kv_heads
        // xv: [1, kv_size] kv_size = num_kv_heads * head_dim
        for i in 0..self.num_kv_heads {
            let range = i*self.head_dim..(i+1)*self.head_dim;
            let head_xv = Matrix::new(xv.data[range].to_vec(), self.head_dim, 1);
            v_cache[i].concat(head_xv, 0);
        }
    }


    pub fn forward(&self, 
        new_input: &Matrix,
        position: usize,
        k_cache: &mut Vec<Matrix>,
        v_cache: &mut Vec<Matrix>
    ) -> Matrix {

        // new_input [1, hidden_size]

        let mut xq = self.q_proj.forward(new_input); // [1, q_size]
        let mut xk = self.k_proj.forward(new_input); // [1, kv_size]
        let xv = self.v_proj.forward(new_input); // [1, kv_size]

        // add rope
        Self::_apply_rope(&mut xq, position, self.num_heads, self.head_dim);
        Self::_apply_rope(&mut xk, position, self.num_kv_heads, self.head_dim);

        // add kvcache
        // k_cache: [head_dim, seq_len] * num_kv_heads
        self._add_to_k_cache(k_cache, xk);
        // v_cache: [seq_len, head_dim] * num_kv_heads
        self._add_to_v_cache(v_cache, xv);

        xq.scale_by(self.scaling);
        let chunked_xq = Self::_chunked_xq_by_heads(xq, self.head_dim);
        // attention
        let mut output = Matrix::new_empty(1, 0); // output will be in shape: [1, q_size]
        for i in 0..self.num_heads {
            // current head
            let q = &chunked_xq[i]; // q: [1, head_dim]
            let _i = i / self.num_queries_per_kv; // share heads with multiple q
            let k = &k_cache[_i]; // k: [head_dim, seq_len]
            let v = &v_cache[_i]; // v: [seq_len, head_dim]

            // score
            let mut score = matmul(&q, &k); // [1, seq_len]

            // soft capping
            score.scale_by(1.0/self.attn_logit_softcapping);
            score.tanh();
            score.scale_by(self.attn_logit_softcapping);

            // score softmax
            score.softmax();
            
            let head_output = matmul(&score, &v);
            output.concat(head_output, 1);
        }

        assert!(output.n_col == self.q_size); // output [1, q_size]

        self.o_proj.forward(&output) // [1, hidden_size]
    }


}


struct GemmaDecoderLayer {
    input_layernorm: RMSNorm,
    self_attn: GemmaAttention,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    mlp: GemmaMLP,
    post_feedforward_layernorm: RMSNorm,
    k_cache: Vec<Matrix>,
    v_cache: Vec<Matrix>
}

impl GemmaDecoderLayer {
    fn _get_next_chunk(weight_data: &Vec<f32>, cursor:&mut usize, step:usize) -> Vec<f32> {
        let res = weight_data[*cursor..*cursor+step].to_vec();
        *cursor += step;
        res
    }
    pub fn new(
        weight_data: Vec<f32>
    ) -> GemmaDecoderLayer {
        let mut cursor = 0;

        let input_layernorm_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 2304);
        let input_layernorm = RMSNorm::new(
            input_layernorm_weight_data,
            2304 // dim = hidden_size
        );
        let self_attn_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 14155776); // 2*hidden_size*(q_size+kv_size) = 2*2304*(8*256+4*256)
        let self_attn = GemmaAttention::new(
            self_attn_weight_data,
            8, // num_heads
            4, // num_kv_heads
            256, // head_dim
            256, // query_pre_attn_scalar
            2304, // hidden_size
            50.0 // attn_logit_softcapping
        );
        let post_attention_layernorm_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 2304);
        let post_attention_layernorm = RMSNorm::new(
            post_attention_layernorm_weight_data,
            2304 // dim = hidden_size
        );
        let pre_feedforward_layernorm_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 2304);
        let pre_feedforward_layernorm = RMSNorm::new(
            pre_feedforward_layernorm_weight_data,
            2304 // dim = hidden_size
        );
        let mlp_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 63700992); // 3*hidden_size*intermediate_size = 3*2304*9216
        let mlp = GemmaMLP::new(
            mlp_weight_data,
            2304, // hidden_size,
            9216 // intermediate_size
        );
        let post_feedforward_layernorm_weight_data = Self::_get_next_chunk(&weight_data, &mut cursor, 2304);
        let post_feedforward_layernorm = RMSNorm::new(
            post_feedforward_layernorm_weight_data,
            2304 // dim = hidden_size
        );

        let mut k_cache:Vec<Matrix> = vec![]; // [head_dim, seq_len] * num_kv_heads = [256,0]*8
        let mut v_cache:Vec<Matrix> = vec![]; // [seq_len, head_dim] * num_kv_heads = [0,256]*8
        for _i in 0..4 {
            let init_k = Matrix::new_empty(256, 0);
            k_cache.push(init_k);
            let init_v = Matrix::new_empty(0, 256);
            v_cache.push(init_v);
        }
        GemmaDecoderLayer {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            mlp,
            post_feedforward_layernorm,
            k_cache,
            v_cache
        }
    }

    pub fn forward(&mut self, new_input:&Matrix, position:usize) -> Matrix {
        // new_input [1, hidden_size]
        let attn_residual = new_input.clone();
        let normed_input = self.input_layernorm.forward(&new_input);
        let hidden_output = self.self_attn.forward(&normed_input , position, &mut self.k_cache, &mut self.v_cache);
        let mut normed_attn_output = self.post_attention_layernorm.forward(&hidden_output);
        normed_attn_output.add(&attn_residual);

        let mlp_residual = normed_attn_output.clone();
        let mlp_input = self.pre_feedforward_layernorm.forward(&normed_attn_output);
        let mlp_output = self.mlp.forward(&mlp_input);
        let mut normed_mlp_output = self.post_feedforward_layernorm.forward(&mlp_output);
        normed_mlp_output.add(&mlp_residual);

        normed_attn_output // output [1, hidden_size]
    }

}

struct GemmaModel {
    layers: Vec<GemmaDecoderLayer>,
    norm: RMSNorm
}

impl GemmaModel {
    pub fn from_fp(fp: &mut File) -> GemmaModel {
        let mut layers = vec![];
        for _i in 0..26 { // num_hidden_layers = 26
            let weight_data = read_next_n_fp32(fp, 77865984).unwrap();
            let layer = GemmaDecoderLayer::new(weight_data);
            layers.push(layer);
        }
        let norm_weight_data = read_next_n_fp32(fp, 2304).unwrap();
        let norm = RMSNorm::new(norm_weight_data, 2304); // hidden_size = 2304
        GemmaModel {
            layers,
            norm
        }
    }

    pub fn forward(&mut self, input:&Matrix, position:usize) -> Matrix {
        let mut hidden_state = input.clone();
        for layer in self.layers.iter_mut() {
            hidden_state = layer.forward(&hidden_state, position);
        }
        self.norm.forward(&hidden_state)
    }

}

struct Gemma2ForCausalLM {
    tokenizer: Tokenizer,
    embedder: Embedding,
    model: GemmaModel,
    sampler: Sampler
}

impl Gemma2ForCausalLM {
    fn _read_and_skip_head(fp: &mut File) {
        let len = fp.read_u32::<LittleEndian>().unwrap();
        let mut str_vec = Vec::with_capacity(len as usize);
        fp.read(&mut str_vec).unwrap();
        let head = String::from_utf8(str_vec).unwrap();
        assert!(&head[..4] == "GRMD", "The model file is not in gemmars format.");
        info!("model type: {}", &head[5..]);
    }
    pub fn new(model_path:&str, tokenizer_path:&str) -> Gemma2ForCausalLM {
        let tokenizer = Tokenizer::from_file(tokenizer_path);

        let mut fp = File::open(model_path).expect(&format!("Cannot read file from {}", model_path));

        Self::_read_and_skip_head(&mut fp);
        
        let embedder = Embedding::from_fp(&mut fp);

        let model = GemmaModel::from_fp(&mut fp);

        let sampler = Sampler::new(&embedder);

        Gemma2ForCausalLM {
            tokenizer,
            embedder,
            model,
            sampler
        }
    }

    // TODO: forward

    // TODO: generate
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_linear() {
        let input = Matrix::new(vec![1.0;1024*3],1024, 3);
        let l = Linear::new(vec![3.14;15], 3, 5);
        let output = l.forward(&input);
        assert_eq!(output.get(300, 4), 9.42);
    }
}
