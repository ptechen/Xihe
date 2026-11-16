//! Xihe 桌面终端入口。

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_title("Xihe"),
        ..Default::default()
    };
    eframe::run_native(
        "xihe",
        options,
        Box::new(|cc| Ok(Box::new(xihe_desktop::ui::XiheApp::new(cc.egui_ctx.clone())))),
    )
}