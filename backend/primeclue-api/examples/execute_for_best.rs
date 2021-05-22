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

use primeclue::data::data_set::DataSet;
use primeclue::data::outcome::Class;
use primeclue::data::{Input, Outcome, Point};
use primeclue::exec::score::Objective::Auc;
use primeclue::exec::tree::Tree;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::ops::Add;
use std::time::{Duration, Instant};

// Run with:
// cargo run --release --example execute_for_best
fn main() {
    let data = get_data().into_view();
    let end = Instant::now().add(Duration::from_secs(10));
    let mut counter = 0;
    let mut best = 0.0;
    while Instant::now().lt(&end) {
        counter += 1;
        for _ in 0..1_000 {
            let tree = Tree::new(data.input_shape(), 30, &[], 0.5, 0.9);
            if let Some(score) = tree.execute_for_score(&data, Class::new(0), &Auc) {
                if score.value() > best {
                    best = score.value();
                }
            }
        }
    }
    println!("Count: {}, best: {}", counter, best);
}

fn get_data() -> DataSet {
    let mut classes = HashMap::new();
    classes.insert(Class::new(0), "A".to_owned());
    classes.insert(Class::new(1), "B".to_owned());
    classes.insert(Class::new(2), "C".to_owned());
    classes.insert(Class::new(3), "D".to_owned());
    let string_classes = classes.iter().map(|(c, s)| (s.clone(), *c)).collect::<HashMap<_, _>>();
    let mut data_set = DataSet::new(classes);
    let mut rng = thread_rng();
    for _ in 0..4 * 250 {
        let a = rng.gen_range(0..100);
        let b = rng.gen_range(0..100);
        let c = rng.gen_range(0..100);

        let output = if a % 15 == 0 {
            "A"
        } else if (b + 2) % 5 == 0 {
            "B"
        } else if (c + 5) % 3 == 0 {
            "C"
        } else {
            "D"
        };

        let point = Point::new(
            Input::from_vector(vec![vec![a as f32, b as f32, c as f32]]).unwrap(),
            Outcome::new(*string_classes.get(output).unwrap(), 1.0, -1.0),
        );
        data_set.add_data_point(point).unwrap();
    }
    data_set
}
