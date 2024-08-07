
#[path = "./function.rs"]
mod function;
use core::num;
use std::io::SeekFrom;

use function::*;

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
    pub fn forward(self, x:&mut Matrix) -> Matrix {
        self._norm(x);
        matmul(x, &self.weight)
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