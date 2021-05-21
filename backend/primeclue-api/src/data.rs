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

use crate::executor::{Status, StatusCallback};
use primeclue::data::data_set::DataSet;
use primeclue::data::importer::{
    build_data_set, class_producer, split_to_vec, ClassRequest, ClassResponse,
};
use primeclue::error::PrimeclueErr;
use primeclue::user::{read_files, Settings, DATA_DIR, DELETE_IN_PROGRESS};
use std::path::Path;
use std::{fs, thread};

pub(crate) fn remove(name: &str) -> Result<(), PrimeclueErr> {
    let settings = Settings::new()?;
    let org_path = Path::new(settings.base_dir()).join(DATA_DIR).join(name);
    let remove_path = Path::new(settings.base_dir())
        .join(DATA_DIR)
        .join(format!("{}_{}", DELETE_IN_PROGRESS, name));
    if remove_path.exists() {
        fs::remove_dir_all(&remove_path)
            .map_err(|e| format!("Unable to remove temporary dir {:?}: {:?}", remove_path, e))?;
    }
    fs::rename(&org_path, remove_path.clone()).map_err(|e| {
        format!("Unable to rename to temporary dir {:?} -> {:?}: {:?}", org_path, remove_path, e)
    })?;
    thread::spawn(move || {
        if let Err(e) = fs::remove_dir_all(&remove_path) {
            println!("Unable to remove temporary dir {:?}: {:?}", remove_path, e);
        }
    });
    Ok(())
}

pub(crate) fn list() -> Result<Vec<String>, PrimeclueErr> {
    let settings = Settings::new()?;
    let data_path = settings.data_dir();
    let rd = fs::read_dir(&data_path)?;
    read_files(rd)
}

pub(crate) fn import(
    r: ClassRequest,
    status_callback: &StatusCallback,
) -> Result<String, PrimeclueErr> {
    let data = build_data_set(&r)?;
    let settings = Settings::new()?;
    let total = data.len();
    let callback = |count| {
        status_callback(Status::Progress(
            count as f64 / total as f64,
            format!("Saved data set {}", count),
        ));
        Ok(())
    };
    save_data(r.data_name, &data, &settings, callback)?;
    Ok("Done".to_string())
}

pub(crate) fn classes(r: &ClassRequest) -> Result<ClassResponse, PrimeclueErr> {
    let data = split_to_vec(&r.content, &r.separator, r.ignore_first_row);
    let mut classes = Vec::with_capacity(data.len());
    let class_producer = class_producer(r, &data)?;
    if r.ignore_first_row {
        classes.push(String::new());
    }
    for row in 0..data.len() {
        let class = class_producer.class(&data, row)?;
        classes.push(class.map_or(String::new(), |c| c.to_string()))
    }
    Ok(ClassResponse::new(classes))
}

fn save_data<F>(
    name: String,
    data: &DataSet,
    settings: &Settings,
    callback: F,
) -> Result<(), PrimeclueErr>
where
    F: FnMut(usize) -> Result<(), PrimeclueErr>,
{
    let path = Path::new(settings.base_dir()).join(DATA_DIR).join(name);
    data.save_to_disk(&path, callback)
}
