"""Save one figure containing every model's training and validation loss."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ("#1F77B4", "#D55E00", "#009E73", "#CC79A7", "#856404",
          "#56B4E9", "#7F3C8D")


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def plot_loss_curves(history, output_path):
    """Plot all models in two panels and save a report-ready PNG."""
    required = {"model", "epoch", "train_mse", "validation_mse"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Loss history is missing columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("Loss history is empty")

    scale = 2
    width, height = 1800 * scale, 900 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(26 * scale)
    heading_font = _font(20 * scale)
    label_font = _font(15 * scale)
    tick_font = _font(13 * scale)
    draw.text((width // 2, 55 * scale), "Mean squared error by training epoch",
              fill="#202124", font=title_font, anchor="mm")

    models = list(history["model"].drop_duplicates())
    grouped = {
        name: history.loc[history["model"] == name].sort_values("epoch")
        for name in models
    }
    max_epoch = float(history["epoch"].max())
    panels = (
        ("Training loss", "train_mse", (130, 145, 850, 655)),
        ("Validation loss", "validation_mse", (1010, 145, 1730, 655)),
    )

    for heading, column, box in panels:
        left, top, right, bottom = (coordinate * scale for coordinate in box)
        values = history[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.08, 1.0)
        y_min = max(0.0, low - padding)
        y_max = high + padding
        draw.text(((left + right) // 2, top - 42 * scale), heading,
                  fill="#202124", font=heading_font, anchor="mm")

        for tick in range(6):
            y = bottom - (bottom - top) * tick / 5
            value = y_min + (y_max - y_min) * tick / 5
            draw.line((left, y, right, y), fill="#E4E7EB", width=scale)
            draw.text((left - 12 * scale, y), f"{value:.0f}",
                      fill="#4B5563", font=tick_font, anchor="rm")
        for tick in range(6):
            x = left + (right - left) * tick / 5
            epoch = max_epoch * tick / 5
            draw.text((x, bottom + 17 * scale), f"{epoch:.0f}",
                      fill="#4B5563", font=tick_font, anchor="mt")
        draw.line((left, top, left, bottom, right, bottom),
                  fill="#374151", width=2 * scale)
        draw.text(((left + right) // 2, bottom + 58 * scale), "Epoch",
                  fill="#202124", font=label_font, anchor="mm")
        draw.text((left - 62 * scale, top - 13 * scale), "MSE",
                  fill="#202124", font=label_font, anchor="mm")

        for index, name in enumerate(models):
            color = COLORS[index % len(COLORS)]
            model_history = grouped[name]
            points = [
                (left + (right - left) * float(epoch) / max_epoch,
                 bottom - (bottom - top) * (float(loss) - y_min) / (y_max - y_min))
                for epoch, loss in zip(model_history["epoch"], model_history[column])
            ]
            if len(points) > 1:
                draw.line(points, fill=color, width=3 * scale, joint="curve")
            else:
                x, y = points[0]
                draw.ellipse((x - 3 * scale, y - 3 * scale,
                              x + 3 * scale, y + 3 * scale), fill=color)

    legend_spacing = 300 * scale
    for index, name in enumerate(models):
        row, column = divmod(index, 5)
        row_count = min(5, len(models) - row * 5)
        legend_start = (width - legend_spacing * (row_count - 1)) // 2
        x = legend_start + column * legend_spacing
        legend_y = (775 + row * 45) * scale
        color = COLORS[index % len(COLORS)]
        draw.line((x - 88 * scale, legend_y, x - 52 * scale, legend_y),
                  fill=color, width=4 * scale)
        draw.text((x - 43 * scale, legend_y), name, fill="#202124",
                  font=label_font, anchor="lm")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width // scale, height // scale), Image.Resampling.LANCZOS).save(
        output_path,
    )
    return output_path
