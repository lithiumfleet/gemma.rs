
pub struct Matrix {
    data: Vec<f32>,
    n_row: usize,
    n_col: usize
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

}

pub fn matmul(output:&mut Matrix, a:&Matrix, b:&Matrix) {
    assert!(output.n_row == a.n_row && output.n_col == b.n_col, 
        "Can not matmul {}*{} with {}*{} and save into {}*{}", 
        a.n_row, a.n_col, b.n_row, b.n_col, output.n_row, output.n_col);
    for i in 0..a.n_row {
        for j in 0..b.n_col {
            let mut temp_sum = 0.0;
            for k in 0..a.n_col {
                temp_sum += a.get(i, k) * b.get(k, j);
            }
            output.set(i, j, temp_sum);
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
        let mut output = Matrix::new_empty(3, 2);
        matmul(&mut output, &a, &b);
        assert_eq!(output.get(2, 0), 5.0);
    }
    #[test]
    fn test_trans() {
        let mut a = Matrix::new(vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0], 3, 3);
        a.transpose();
        assert_eq!(a.get(2, 1), 6.0);
    }
}