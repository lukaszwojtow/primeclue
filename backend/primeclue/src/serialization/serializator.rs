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

use crate::error::PrimeclueErr;
use crate::serialization::serializable::Serializable;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub const PRIMECLUE_SPACE_SUBSTITUTE: &str = "PRIMECLUE_SPACE_SUBSTITUTE";
pub const SERIALIZED_FILE_EXT: &str = ".ssd";

#[derive(Default, Debug)]
pub struct Serializator {
    strings: Vec<String>,
    next_token: usize,
}

impl Serializator {
    #[must_use]
    pub fn new() -> Self {
        Serializator { strings: vec![], next_token: 0 }
    }

    pub fn next_token(&mut self) -> Result<&String, &str> {
        if let Some(s) = self.strings.get(self.next_token) {
            self.next_token += 1;
            Ok(s)
        } else {
            Err("Not enough tokens")
        }
    }

    pub fn add_items(&mut self, a: &[&dyn Serializable]) {
        a.iter().for_each(|i| self.add(*i));
    }

    pub fn add_str(&mut self, v: &str) {
        let v = v.to_owned();
        self.add_string(v);
    }

    pub fn add_string(&mut self, v: String) {
        let v = v.replace(' ', PRIMECLUE_SPACE_SUBSTITUTE);
        self.strings.push(v);
    }

    pub fn add(&mut self, v: &dyn Serializable) {
        v.serialize(self);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let cs = format!("{} ", self.calc_checksum());
        let mut bytes = cs.into_bytes();
        bytes.extend_from_slice(self.as_serialized().as_bytes());
        bytes
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let content = String::from_utf8(bytes)
            .map_err(|e| format!("Cannot convert bytes to String {}", e))?;
        let (read_check_sum, content) = Serializator::extract_check_sum(&content)?;
        let strings: Vec<String> = content.split_whitespace().map(ToOwned::to_owned).collect();
        let serializator = Serializator { strings, next_token: 0 };
        let actual_check_sum = serializator.calc_checksum();
        if read_check_sum == actual_check_sum {
            Ok(serializator)
        } else {
            Err("Invalid checksum".to_string())
        }
    }

    pub fn save(&self, dir: &Path, name: &str) -> Result<usize, PrimeclueErr> {
        if !Path::new(&dir).exists() {
            return PrimeclueErr::result(format!("Directory {:?} does not exist", dir));
        }
        let mut f = File::create(dir.join(name))
            .map_err(|e| format!("Unable to create file {:?}: {}", name, e))?;
        let bytes = self.to_bytes();
        let bytes_written =
            f.write(&bytes).map_err(|e| format!("Unable to write to file: {}", e))?;
        Ok(bytes_written)
    }

    pub fn load(path: &Path) -> Result<Serializator, String> {
        if path.exists() {
            let bytes =
                fs::read(path).map_err(|e| format!("Unable to read file, error: {:?}", e))?;
            Serializator::from_bytes(bytes)
        } else {
            Err(format!("Path {:?} does not exists", path))
        }
    }

    fn extract_check_sum(content: &str) -> Result<(u128, String), String> {
        let first = content.find(' ').ok_or("Unable to find check sum")?;
        let (cs, rest) = content.split_at(first);
        let check_sum: u128 =
            cs.parse().map_err(|e| format!("Unable to parse checksum: {}", e))?;
        Ok((check_sum, rest.to_owned()))
    }

    #[must_use]
    pub fn as_serialized(&self) -> String {
        let mut all = String::from("");
        for string in &self.strings {
            let with_space = format!("{} ", string);
            all.push_str(&with_space);
        }
        all
    }

    fn calc_checksum(&self) -> u128 {
        const LIMIT: u128 = u128::MAX >> 7;
        let mut cs: u128 = 1;
        for string in &self.strings {
            for byte in string.as_bytes() {
                if cs > LIMIT {
                    cs >>= 8;
                }
                cs *= u128::from(*byte) + 1;
            }
        }
        cs
    }
}

#[cfg(test)]
pub mod test {

    #[derive(Debug, PartialEq)]
    struct Test {
        s: String,
        n: usize,
    }

    impl Serializable for Test {
        fn serialize(&self, s: &mut Serializator) {
            s.add(&self.n);
            s.add(&self.s);
        }
    }

    impl Deserializable for Test {
        fn deserialize(s: &mut Serializator) -> Result<Self, String> {
            let n = usize::deserialize(s)?;
            let s = String::deserialize(s)?;
            Ok(Test { s, n })
        }
    }

    use crate::serialization::{Deserializable, Serializable, Serializator};
    use std::collections::HashMap;
    use std::fmt::Debug;

    pub fn test_serialization<T: Serializable + Deserializable + PartialEq + Debug>(v: T) {
        let mut s = Serializator::new();
        s.add(&v);
        let bytes = s.to_bytes();
        let mut d = Serializator::from_bytes(bytes).unwrap();
        let n = T::deserialize(&mut d).unwrap();
        assert_eq!(n, v);
    }

    #[test]
    fn test_bool() {
        let f = false;
        test_serialization(f);
        let t = true;
        test_serialization(t);
    }

    #[test]
    fn test_vec() {
        let v = vec![1u16, 2, 3];
        test_serialization(v);
    }

    #[test]
    fn test_string_with_space() {
        let s = "Some string with spaces".to_owned();
        let n = 43;
        let test = Test { s, n };
        test_serialization(test);
    }

    #[test]
    fn test_map() {
        let mut m = HashMap::new();
        m.insert(1u16, "1".to_string());
        m.insert(2, "2".to_string());
        m.insert(3, "3".to_string());
        test_serialization(m);
    }

    #[test]
    fn test_tuple() {
        let mut m = HashMap::new();
        m.insert(0usize, 0.0);
        m.insert(1usize, 1.0);
        let t = (vec!["one".to_string(), "two".to_string(), "free".to_string()], m);
        test_serialization(t);
    }

    #[test]
    fn test_enough_tokens() {
        let t = Test { s: "Some text".to_string(), n: 42 };
        let mut s = Serializator::new();
        s.add(&t);
        s.strings.remove(s.strings.len() - 1);
        let r = Test::deserialize(&mut s);
        assert!(r.is_err());
        assert_eq!(r.err().unwrap(), "Not enough tokens");
    }

    #[test]
    fn test_invalid_checksum() {
        let t = Test { s: "Some text".to_string(), n: 42 };
        let mut s = Serializator::new();
        s.add(&t);
        let mut bytes = s.to_bytes();
        bytes[60] = 15;
        let r = Serializator::from_bytes(bytes);
        assert!(r.is_err());
        assert_eq!(r.err().unwrap(), "Invalid checksum");
    }

    #[test]
    fn test_option_invalid_token() {
        let mut s = Serializator::new();
        let o = Some(15usize);
        s.add(&o);
        let index = s.strings.len() - 2;
        s.strings[index] = "Smoe".to_owned();
        let bytes = s.to_bytes();
        let mut s = Serializator::from_bytes(bytes).unwrap();
        let r: Result<Option<usize>, String> = Option::deserialize(&mut s);
        assert!(r.is_err());
        assert_eq!(r.err().unwrap(), "Invalid token when deserializing Option: 'Smoe'");
    }
}
