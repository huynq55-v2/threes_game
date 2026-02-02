use pyo3::prelude::*;

pub mod game;
pub mod pseudo_list;
pub mod python_module;
pub mod rarity;
pub mod threes_const;
pub mod tile;
pub mod ui;

#[pymodule]
fn threes_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_module::ThreesEnv>()?;
    m.add_class::<python_module::ThreesVecEnv>()?;
    Ok(())
}
