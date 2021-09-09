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

use rand::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::env;

lazy_static! {
    pub static ref GET_RNG: fn() -> Box<dyn RngCore> = if env::var("USE_PREDICTABLE_RNG").is_ok()
    {
        eprintln!("\x1b[0;31mWARNING: Using predictable RNG\x1b[0m");
        predictable_rng
    } else {
        thread_rng
    };
}

fn thread_rng() -> Box<dyn RngCore> {
    Box::new(rand::thread_rng())
}

fn predictable_rng() -> Box<dyn RngCore> {
    Box::new(XorShiftRng::seed_from_u64(42))
}

#[cfg(test)]
mod test {
    use crate::contrand::predictable_rng;
    use rand::Rng;

    #[test]
    fn test_predictable_rng() {
        let mut rng1 = predictable_rng();
        let mut rng2 = predictable_rng();
        for _ in 0..1024 {
            assert_eq!(rng1.gen::<usize>(), rng2.gen::<usize>());
            assert!((rng1.gen::<f64>() - rng2.gen::<f64>()).abs() < f64::EPSILON);
            assert_eq!(rng1.gen_range(0..1024), rng2.gen_range(0..1024));
            assert_eq!(rng1.gen_bool(0.3), rng2.gen_bool(0.3));
        }
    }
}
