// SPDX-License-Identifier: AGPL-3.0
/*
   Primeclue: Machine Learning and Data Mining
   Copyright (C) 2020 Łukasz Wojtów

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use primeclue::data::data_set::DataSet;
use primeclue::data::outcome::Class;
use primeclue::data::{Input, InputShape, Outcome, Point};
use primeclue::exec::class_training::ClassTraining;
use primeclue::exec::score::Objective::{Auc, Cost};
use primeclue::exec::training_group::TrainingGroup;
use primeclue::exec::tree::Tree;
use primeclue::rand::GET_RNG;
use rand::Rng;
use std::collections::HashMap;

fn select_node(c: &mut Criterion) {
    let mut rng = GET_RNG();
    let mut trees = vec![];
    for _ in 0..100 {
        let max_depth = rng.gen_range(2, 10);
        let data_prob = rng.gen_range(0.01, 0.99);
        let branch_prob = rng.gen_range(0.01, 0.99);
        let tree = Tree::new(&InputShape::new(1, 10), max_depth, &[], branch_prob, data_prob);
        trees.push(tree)
    }

    c.bench_function("select_node", |b| {
        b.iter(|| {
            for tree in trees.iter_mut() {
                let n = tree.select_random_node();
                black_box(n);
            }
        })
    });
}

fn check_nans_same(c: &mut Criterion) {
    let vec: Vec<f64> = vec![42.42; 100_000];

    c.bench_function("check_nans_same", |b| {
        b.iter(|| {
            let mut changed = false;
            if !vec.iter().any(|f| !f.is_finite()) {
                for v in &vec {
                    if !v.is_finite() {
                        break;
                    }
                    changed = changed || (*v - vec[0]).abs() > 0.001;
                }
                black_box(changed);
            }
        })
    });
}

fn threshold_cost_bench(c: &mut Criterion) {
    let data = create_sample_data(10_000);
    let mut rng = GET_RNG();
    let mut outcomes = vec![];
    for p in data.iter() {
        outcomes.push((rng.gen_range(0.0, 1.0), p.data().1.clone()));
    }
    c.bench_function("threshold_cost", |b| {
        b.iter(|| {
            let t = Cost.threshold(&outcomes, Class::new(1));
            black_box(t);
        })
    });
}

fn training_group_generation_bench(c: &mut Criterion) {
    let (training_data, verification_data, _) =
        create_sample_data(1_000).shuffle().into_3_views_split();
    let mut training_group =
        TrainingGroup::new(training_data, verification_data, &Auc, 10, &[]).unwrap();
    c.bench_function("training_group_generation_bench", |b| {
        b.iter(|| {
            training_group.next_generation();
            black_box(training_group.generation());
        })
    });
}

fn vec_add_fast(c: &mut Criterion) {
    let data = create_sample_data(1_000).into_view();
    c.bench_function("vec_add_fast", |b| {
        b.iter(|| {
            let mut col1 = data.cells().get(0, 0).clone();
            let col2 = data.cells().get(0, 1);
            for _ in 0..1000 {
                // col1.iter_mut().enumerate().for_each(|(i, x)| *x = *x + col2[i]);// slower, probably due to [] access check
                // col1.iter_mut().zip(col2).for_each(|(c1, c2)| *c1 = *c1 + *c2); // the same as below, but less readable
                for (value1, value2) in col1.iter_mut().zip(col2) {
                    *value1 = *value1 + *value2;
                }
            }
            black_box(col2);
        })
    });
}

fn next_generation_bench(c: &mut Criterion) {
    let (training_data, verification_data, _) =
        create_sample_data(1_000).shuffle().into_3_views_split();

    let mut class_training = ClassTraining::new(10, vec![], &Auc, Class::new(0));
    c.bench_function("next_generation", |b| {
        b.iter(|| {
            class_training.next_generation(black_box(&training_data), &verification_data);
        })
    });
}

fn execute_tree_bench(c: &mut Criterion) {
    let data = create_sample_data(1_000).into_view();
    let mut rng = GET_RNG();
    let mut trees = vec![];
    for _ in 0..100 {
        let max_depth = rng.gen_range(2, 10);
        let data_prob = rng.gen_range(0.01, 0.99);
        let branch_prob = rng.gen_range(0.01, 0.99);
        let tree = Tree::new(data.input_shape(), max_depth, &[], branch_prob, data_prob);
        trees.push(tree)
    }

    c.bench_function("execute_tree", |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = tree.execute_for_score(black_box(&data), Class::new(0), &Auc);
            }
        })
    });
}

fn create_tree_bench(c: &mut Criterion) {
    let forbidden_cols = vec![1, 2, 3];
    c.bench_function("create_tree", |b| {
        b.iter(|| {
            let mut rng = GET_RNG();
            for _ in 0..100 {
                let max_depth = rng.gen_range(2, 10);
                let data_prob = rng.gen_range(0.01, 0.99);
                let branch_prob = rng.gen_range(0.01, 0.99);
                let _ = Tree::new(
                    black_box(&InputShape::new(1, 6)),
                    max_depth,
                    &forbidden_cols,
                    branch_prob,
                    data_prob,
                );
            }
        })
    });
}

criterion_group!(
    benches,
    execute_tree_bench,
    create_tree_bench,
    next_generation_bench,
    training_group_generation_bench,
    select_node,
    check_nans_same,
    vec_add_fast,
    threshold_cost_bench
);
criterion_main!(benches);

fn create_sample_data(count: usize) -> DataSet {
    let mut classes = HashMap::new();
    classes.insert(Class::new(0), "false".to_string());
    classes.insert(Class::new(1), "true".to_string());
    let mut ds = DataSet::new(classes);
    for _ in 0..count {
        ds.add_data_point(Point::new(
            Input::from_vector(vec![vec![1.0, 2.0]]).unwrap(),
            Outcome::new(Class::new(0), 1.0, -1.0),
        ))
        .unwrap();
        ds.add_data_point(Point::new(
            Input::from_vector(vec![vec![2.0, 3.0]]).unwrap(),
            Outcome::new(Class::new(1), 1.0, -1.0),
        ))
        .unwrap();
    }
    ds
}
