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
use std::fs;
use std::fs::{DirEntry, ReadDir};
use std::path::{Path, PathBuf};

pub const DELETE_IN_PROGRESS: &str = "delete_in_progress";
pub const DATA_DIR: &str = "data";
pub const CLASSIFIERS_DIR: &str = "classifiers";

#[derive(Clone, Debug)]
pub struct Settings {
    base_dir: PathBuf,
}

impl Settings {
    pub fn new() -> Result<Settings, String> {
        // TODO get base directory from env variable if present
        let base_dir = dirs::home_dir()
            .ok_or_else(|| "Unable to get user's home directory".to_string())?
            .join("Primeclue");
        create_dir(&base_dir)?;
        let data = base_dir.join(DATA_DIR);
        create_dir(&data)?;
        let projects = base_dir.join(CLASSIFIERS_DIR);
        create_dir(&projects)?;
        Ok(Settings { base_dir })
    }

    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }

    pub fn data_dir(&self) -> PathBuf {
        self.base_dir.clone().join(DATA_DIR)
    }

    pub fn classifier_dir(&self) -> PathBuf {
        self.base_dir.clone().join(CLASSIFIERS_DIR)
    }
}

fn create_dir(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        fs::create_dir_all(dir)
            .map_err(|e| format!("Unable to create directory {:?}: {}", dir, e))
    } else if dir.is_dir() {
        Ok(())
    } else {
        Err(format!("{:?} is not a directory", dir))
    }
}

pub fn read_files(list: ReadDir) -> Result<Vec<String>, PrimeclueErr> {
    let mut projects = vec![];
    for file in list {
        let file = file?;
        let name = file.file_name();
        if empty(&file) {
            continue;
        }
        let utf_name = name
            .to_str()
            .ok_or_else(|| PrimeclueErr::from(format!("Unable to read file: {:?}", file)))?
            .to_owned();
        if !utf_name.contains(DELETE_IN_PROGRESS) {
            projects.push(utf_name);
        }
    }
    projects.sort();
    Ok(projects)
}

fn empty(name: &DirEntry) -> bool {
    match fs::read_dir(name.path()) {
        Ok(mut list) => list.next().is_none(),
        Err(_) => true,
    }
}
