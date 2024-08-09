
#[path = "./function.rs"]
mod function;

use function::*;
use serde_json::to_vec;

struct Linear {
    weight_t: Matrix,
    in_features: usize,
    out_features: usize
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

    pub fn forward(self, x:&Matrix) -> Matrix {
        let mut gate = self.gate_proj.forward(x);
        gelu(&mut gate);
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

    pub fn forward(self, input_ids:Vec<u32>) -> Matrix {
        let mut output = vec![];
        for i in 0..input_ids.len() {
            output.push(self.weight.data[i]);
        }
        Matrix::new(output, input_ids.len(), self.embedding_dim)
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
    pub fn forward(&self, x:&mut Matrix) -> Matrix {
        self._norm(x);
        matmul(x, &self.weight)
    }
}


struct GemmaAttention { // this is GQA
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

    fn _get_chunked_q(xq:Matrix, head_dim:usize) -> Vec<Matrix> {
        assert!(xq.n_row == 1 && xq.n_col % head_dim == 0);
        let mut chunked_q:Vec<Matrix> = vec![];
        for i in 0..xq.data.len() {
            chunked_q.push(
                Matrix::new(xq.data[i*head_dim..(i+1)*head_dim].to_vec(), 1, head_dim)
            );
        }
        chunked_q
    }

    fn _get_repeated_k(ori_k:&Matrix, times:usize) -> Matrix {
        let mut k = Matrix::new(ori_k.data.clone(), ori_k.n_row, ori_k.n_col);
        for _ in 1..times {
            let _k = Matrix::new(ori_k.data.clone(), ori_k.n_row, ori_k.n_col);
            k.concat(_k, 0);
        }
        k
    }
    
    fn _apply_rope(x: &mut Matrix, pos: usize, num_heads:usize, head_dim: usize) {
        assert!(x.n_row == 1 && x.n_col == num_heads*head_dim);

        for i in 0..num_heads {
            let head_offset = i*head_dim;

            for j in 0..head_dim/2 {
                let theta = (pos as f32) / (10000.0_f32.powf(2.0 * j as f32 / head_dim as f32));
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = x.data[2*j+head_offset];
                let x1 = x.data[2*j+1+head_offset];

                x.data[2*j+head_offset] = cos_theta*x0 - sin_theta*x1;
                x.data[2*j+1+head_offset] = sin_theta*x0 + cos_theta*x1;
            }
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
        Self::_apply_rope(&mut xk, position, self.num_kv_heads, self.kv_size);

        xk.transpose(); // [kv_size, 1]
        xq.scale_by(self.scaling);

        // add kv cache
        let chunked_q = Self::_get_chunked_q(xq, self.head_dim); // [1, head_dim] * num_heads
        k_cache.push(xk); // [kv_size, seq_len] * num_heads
        v_cache.push(xv); // [seq_len, kv_size] * num_heads


        // attention
        let mut output = Matrix::new_empty(1, 0); // [1, hidden_size]
        for i in 0..self.num_heads {
            // current head
            let q = &chunked_q[i]; // q: [1, head_dim]
            let k = Self::_get_repeated_k(&k_cache[i], self.num_queries_per_kv); // [head_dim, seq_len]

            let mut score = matmul(&q, &k); // [1, seq_len]
            // score softmax
            score.softmax();

            let v = &v_cache[i]; // [seq_len, kv_size]
            let head_output = matmul(&score, &v);
            output.concat(head_output, 1);
        }
        assert!(output.n_col == self.hidden_size);
        output 

        // output [1, hidden_size]
    }


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