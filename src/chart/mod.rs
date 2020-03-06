use std::error::Error;
use std::path::Path;

use ndarray::{Array1, ArrayView, ArrayView1, azip, Zip};
use plotters::coord::Shift;
use plotters::drawing::bitmap_pixel::RGBPixel;
use plotters::prelude::*;

pub struct Chart<'a> {
    root: DrawingArea<BitMapBackend<'a, RGBPixel>, Shift>,
}


impl<'a> Chart<'a> {
    pub fn bitmap<T: AsRef<Path> + ?Sized>(path: &'a T, (w, h): (u32, u32)) -> Self {
        Self::new(BitMapBackend::new("plotters-doc-data/0.png", (640, 480)).into_drawing_area())
    }

    pub fn new(root: DrawingArea<BitMapBackend<'a, RGBPixel>, Shift>) -> Self {
        root.fill(&WHITE);

        Chart { root }
    }

    pub fn scatter(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<(), Box<dyn Error>> {
        let mut chart = ChartBuilder::on(&self.root)
            // .caption("Sine and Cosine", ("sans-serif", (10).percent_height()))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_ranged(0f64..1.5f64, 0f64..1.5f64)?;

        chart.configure_mesh().draw()?;

        let mut series = Vec::new();
        azip!((&x in &x, &y in &y) series.push(Circle::new((x, y), 2, RED.filled())));

        chart.draw_series(
            series
        )?;

        Ok(())
    }

    pub fn present(&self) {
        self.root.present();
    }
}