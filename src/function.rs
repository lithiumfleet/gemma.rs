use std::f32::consts::PI;


pub struct Matrix {
    pub data: Vec<f32>,
    pub n_row: usize,
    pub n_col: usize
}

impl Matrix {
    pub fn new(data: Vec<f32>, n_row: usize, n_col: usize) -> Self {
        assert_eq!(data.len(), n_row * n_col, "Data length does not match dimensions");
        Matrix {
            data,
            n_row,
            n_col,
        }
    }

    pub fn new_empty(n_row: usize, n_col: usize) -> Self {
        Matrix {
            data: vec![0.0; n_col*n_row],
            n_row,
            n_col,
        }
    }

    pub fn transpose(&mut self) {
        let mut new_data:Vec<f32> = vec![0.0; self.data.len()];
        for r in 0..self.n_row {
            for c in 0..self.n_col {
                new_data[c*self.n_row+r] = self.get(r, c);
            }
        }
        self.data = new_data;

        let _temp = self.n_col;
        self.n_col = self.n_row;
        self.n_row = _temp;
    }

    pub fn get(&self, r:usize, c:usize) -> f32 {
        self.data[r*self.n_col+c]
    }

    pub fn set(&mut self, r:usize, c:usize, val:f32) {
        self.data[r*self.n_col+c] = val;
    }

    pub fn concat(&mut self, other:Matrix, axis:usize) {
        assert!(axis == 0 || axis == 1);
        if axis == 0 {
            assert!(other.n_col == self.n_col);
            self.data.extend(other.data);
            self.n_row += other.n_row;
        } else {
            assert!(other.n_row == self.n_row);
            self.data.extend(other.data);
            self.n_col += other.n_col;
        }
    }

    pub fn scale_by(&mut self, scaler:f32) {
        for i in self.data.iter_mut() {
            *i *= scaler;
        }
    }

    pub fn softmax(&mut self) {
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        
        for val in &mut self.data {
            *val = (*val - max_val).exp();
            sum_exp += *val;
        }

        for val in &mut self.data {
            *val /= sum_exp;
        }
    }

}

pub fn matmul(a:&Matrix, b:&Matrix) -> Matrix {
    assert!(a.n_col == b.n_row, 
        "Can not matmul {}*{} with {}*{}", a.n_row, a.n_col, b.n_row, b.n_col);

    let mut output = Matrix::new_empty(a.n_row, b.n_col);

    for i in 0..a.n_row {
        for j in 0..b.n_col {
            let mut temp_sum = 0.0;
            for k in 0..a.n_col {
                temp_sum += a.get(i, k) * b.get(k, j);
            }
            output.set(i, j, temp_sum);
        }
    }

    output
}

pub fn gelu(x:&mut Matrix) {
    for r in 0..x.n_row {
        for c in 0..x.n_col {
            let i = x.get(r, c);
            let val = 0.5*i*(1.0+f32::tanh((2.0/PI).sqrt()*(i+0.044715*i.powi(3))));
            x.set(r, c, val);
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matmul() {
        let  a  = Matrix::new(vec![1.0,2.0,3.0,4.0,5.0,6.0], 3, 2);
        let  b  = Matrix::new(vec![1.0,-1.0,0.0,2.0], 2, 2);
        let output = matmul(&a, &b);
        assert_eq!(output.get(2, 0), 5.0);
    }
    #[test]
    fn test_trans() {
        let mut a = Matrix::new(vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], 3, 3);
        a.transpose();
        assert_eq!(a.get(2, 1), 6.0);
        assert_eq!(a.get(1, 2), 8.0);
    }
}