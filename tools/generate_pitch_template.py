from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 800
HEIGHT = 520
LINE_COLOR = (245, 245, 245)
GRASS_DARK = (33, 120, 55)
GRASS_LIGHT = (39, 138, 63)
LINE_WIDTH = 4


def create_pitch_template() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), GRASS_DARK)
    draw = ImageDraw.Draw(image)

    stripe_width = 80
    for left in range(0, WIDTH, stripe_width):
        color = GRASS_LIGHT if (left // stripe_width) % 2 == 0 else GRASS_DARK
        draw.rectangle((left, 0, left + stripe_width - 1, HEIGHT), fill=color)

    left, top, right, bottom = 20, 20, 780, 500
    center_x, center_y = WIDTH // 2, HEIGHT // 2

    draw.rectangle((left, top, right, bottom), outline=LINE_COLOR, width=LINE_WIDTH)
    draw.line((center_x, top, center_x, bottom), fill=LINE_COLOR, width=LINE_WIDTH)
    draw.ellipse(
        (center_x - 60, center_y - 60, center_x + 60, center_y + 60),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.ellipse(
        (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
        fill=LINE_COLOR,
    )

    penalty_top, penalty_bottom = 130, 390
    six_top, six_bottom = 195, 325
    left_penalty_x, right_penalty_x = 150, 650
    left_six_x, right_six_x = 75, 725
    left_spot_x, right_spot_x = 95, 705

    draw.rectangle(
        (left, penalty_top, left_penalty_x, penalty_bottom),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.rectangle(
        (right_penalty_x, penalty_top, right, penalty_bottom),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.rectangle(
        (left, six_top, left_six_x, six_bottom),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.rectangle(
        (right_six_x, six_top, right, six_bottom),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )

    draw.ellipse(
        (left_spot_x - 4, center_y - 4, left_spot_x + 4, center_y + 4),
        fill=LINE_COLOR,
    )
    draw.ellipse(
        (right_spot_x - 4, center_y - 4, right_spot_x + 4, center_y + 4),
        fill=LINE_COLOR,
    )
    draw.arc(
        (left_spot_x - 60, center_y - 60, left_spot_x + 60, center_y + 60),
        start=310,
        end=50,
        fill=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.arc(
        (right_spot_x - 60, center_y - 60, right_spot_x + 60, center_y + 60),
        start=130,
        end=230,
        fill=LINE_COLOR,
        width=LINE_WIDTH,
    )

    goal_depth = 12
    goal_height = 90
    draw.rectangle(
        (left - goal_depth, center_y - goal_height // 2, left, center_y + goal_height // 2),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )
    draw.rectangle(
        (right, center_y - goal_height // 2, right + goal_depth, center_y + goal_height // 2),
        outline=LINE_COLOR,
        width=LINE_WIDTH,
    )

    corner_radius = 18
    draw.arc((left - corner_radius, top - corner_radius, left + corner_radius, top + corner_radius), 0, 90, fill=LINE_COLOR, width=LINE_WIDTH)
    draw.arc((right - corner_radius, top - corner_radius, right + corner_radius, top + corner_radius), 90, 180, fill=LINE_COLOR, width=LINE_WIDTH)
    draw.arc((left - corner_radius, bottom - corner_radius, left + corner_radius, bottom + corner_radius), 270, 360, fill=LINE_COLOR, width=LINE_WIDTH)
    draw.arc((right - corner_radius, bottom - corner_radius, right + corner_radius, bottom + corner_radius), 180, 270, fill=LINE_COLOR, width=LINE_WIDTH)

    return image


def save_pitch_template(path: str | Path = "assets/pitch_template.png") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_pitch_template().save(output_path)
    return output_path


if __name__ == "__main__":
    save_pitch_template()
