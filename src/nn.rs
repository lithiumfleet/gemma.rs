
#[path = "./function.rs"]
mod function;
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

    pub fn forward(&self, output:&mut Matrix, input: &Matrix) {
        matmul(output, input, &self.weight_t);
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
    
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_linear() {
        let mut output = Matrix::new_empty(1024, 5);
        let input = Matrix::new(vec![1.0;1024*3],1024, 3);
        let l = Linear::new(vec![3.14;15], 3, 5);
        l.forward(&mut output, &input);
        assert_eq!(output.get(300, 4), 9.42);
    }

}

