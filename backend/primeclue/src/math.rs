pub fn std_dev(values: &[f32]) -> Vec<f32> {
    let avg = values.iter().sum::<f32>() / values.len() as f32;
    let st_dev = (values.iter().map(|v| (v - avg).powf(2.0)).sum::<f32>()
        / (values.len() - 1) as f32)
        .sqrt();
    values.iter().map(|v| (v - avg) / st_dev).collect()
}

pub fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|v1, v2| v1.partial_cmp(&v2).unwrap());
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle] + values[middle - 1]) / 2.0
    } else {
        values[middle]
    }
}

pub fn valid(values: &[f32]) -> bool {
    let mut change = false;
    for v in values {
        if !v.is_finite() {
            return false;
        }
        change = change || (*v - values[0]).abs() > 0.001;
    }
    change
}

#[cfg(test)]
mod test {
    use crate::math::{median, valid};

    #[test]
    fn median_test() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut v), 2.5);
        let mut v = vec![4.0, 3.0, 2.0, 1.0];
        assert_eq!(median(&mut v), 2.5);
        let mut v = vec![4.0, 3.0, 1.0, 2.0];
        assert_eq!(median(&mut v), 2.5);
        let mut v = vec![1.0, 2.0, 3.0];
        assert_eq!(median(&mut v), 2.0);
        let mut v = vec![3.0, 2.0, 1.0];
        assert_eq!(median(&mut v), 2.0);
        let mut v = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut v), 2.0)
    }

    #[test]
    fn test_valid() {
        assert!(!valid(&vec![]));
        let values = vec![-1.0, 2.0, 3.0];
        assert!(valid(&values));

        let values = vec![f32::NAN, -1.0, 2.0, 3.0];
        assert!(!valid(&values));
        let values = vec![-1.0, 2.0, 3.0, f32::NAN];
        assert!(!valid(&values));
        let values = vec![-1.0, f32::NAN, 3.0];
        assert!(!valid(&values));

        let values = vec![f32::INFINITY, -1.0, 2.0, 3.0];
        assert!(!valid(&values));
        let values = vec![-1.0, 3.0, -f32::INFINITY, 54.0];
        assert!(!valid(&values));
        let values = vec![-1.0, 2.0, 3.0, f32::INFINITY];
        assert!(!valid(&values));

        let values = vec![4.0, 4.0, 4.00001];
        assert!(!valid(&values));
    }
}
