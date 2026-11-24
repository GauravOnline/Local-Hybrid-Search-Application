"""Create a reproducible synthetic corpus and draft binary relevance judgments."""

import argparse
import csv
import json
from pathlib import Path

import fitz
from docx import Document
from PIL import Image, ImageDraw, ImageFont

# Each query has one intentionally relevant file; unrelated entries are distractors.
CASES = [
    ("exact", "Orion-482", "Orion-482", "The Orion-482 asset inventory was approved on Monday."),
    ("exact", "ZEPHYR-19", "ZEPHYR-19", "The ZEPHYR-19 maintenance ticket requires a replacement sensor."),
    ("exact", "Invoice 78431", "invoice", "Invoice 78431 covers the annual library subscription."),
    ("exact", "Cedar Protocol", "cedar", "The Cedar Protocol defines emergency building evacuation steps."),
    ("exact", "Atlas backup", "atlas", "The Atlas backup completed successfully at 02:15."),
    ("conceptual", "car accident", "incident", "A vehicle collision occurred at the intersection. Both drivers exchanged insurance details."),
    ("conceptual", "how to save electricity", "energy", "Lower household power consumption by switching off unused appliances and using efficient LED lamps."),
    ("conceptual", "ways to improve sleep", "rest", "A consistent bedtime, a cool bedroom, and limiting caffeine can support better nightly rest."),
    ("conceptual", "protect online accounts", "security", "Enable multi-factor authentication and use unique passwords to prevent unauthorized access."),
    ("conceptual", "cheap vacation planning", "travel", "Compare low-cost flights, share accommodation, and visit free attractions to reduce holiday expenses."),
    ("metadata", "budget_q7", "budget_q7", "Estimated quarterly spending for a community event."),
    ("metadata", "receipt_x41", "receipt_x41", "Purchase record for a replacement keyboard."),
    ("metadata", "meeting_tuesday", "meeting_tuesday", "Agenda includes project milestones and team updates."),
    ("metadata", "garden_2026", "garden_2026", "Watering schedule and soil preparation instructions."),
    ("metadata", "workshop_c12", "workshop_c12", "Attendees should bring their own notebooks."),
    ("ocr", "LUMEN-204", "notice", "LUMEN-204\nConference room reservation\nFriday at 10:30 AM"),
    ("ocr", "MAPLE-735", "capture", "MAPLE-735\nPayment confirmation\nTotal amount: 48 dollars"),
    ("ocr", "NOVA-851", "poster", "NOVA-851\nPhotography exhibit\nMain hall entrance"),
    ("ocr", "EMBER-620", "scan", "EMBER-620\nDelivery instructions\nUse the north entrance"),
    ("ocr", "RIVER-913", "sign", "RIVER-913\nEquipment inspection\nNext review in June"),
]

DISTRACTORS = [
    "A sourdough starter requires flour and water.", "The museum displays ancient pottery.",
    "A violin has four strings and a wooden body.", "Astronomers observed a distant comet.",
    "The watercolour painting uses blue and yellow pigments.", "A chess tournament begins at noon.",
    "The coastal trail follows a rocky shoreline.", "A ceramic bowl was fired in a kiln.",
    "The recipe uses lentils, cumin and tomatoes.", "The piano recital features classical compositions.",
    "A telescope lens needs careful cleaning.", "The knitting pattern uses wool yarn.",
    "The geology lecture addressed sedimentary rocks.", "A fountain pen uses liquid ink.",
    "The sculpture is carved from marble.", "A bicycle chain requires lubrication.",
    "The orchestra rehearses in the auditorium.", "The botanical exhibit features tropical orchids.",
    "A bread recipe calls for yeast and salt.", "The poetry collection describes autumn leaves.",
    "The aquarium houses freshwater fish.", "A pottery class teaches wheel throwing.",
    "The history exhibit traces ancient civilizations.", "The astronomy club studies lunar craters.",
    "A music theory lesson explains major scales.", "A hiking guide describes alpine meadows.",
    "The cookbook includes roasted vegetables.", "A craft project uses coloured paper.",
    "The language lesson practices verb conjugation.", "A birdwatching journal records sparrows.",
    "The fabric sample is woven from cotton.", "The science fair features a volcano model.",
    "The dance class teaches basic steps.", "The art gallery displays abstract landscapes.",
    "A crossword puzzle contains twenty clues.",
]


def write_file(path: Path, text: str) -> None:
    """Write known source content in the selected supported format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".txt":
        path.write_text(text, encoding="utf-8")
    elif path.suffix == ".docx":
        document = Document()
        for line in text.splitlines():
            document.add_paragraph(line)
        document.save(path)
    elif path.suffix == ".pdf":
        with fitz.open() as document:
            page = document.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 545, 760), text, fontsize=16)
            document.save(path)
    else:
        image = Image.new("RGB", (1200, 350), "white")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 40)
        except OSError:
            font = ImageFont.load_default(size=40)
        ImageDraw.Draw(image).multiline_text((45, 45), text, font=font, fill="black", spacing=15)
        image.save(path)


def generate(destination: Path, scale: int = 250) -> None:
    """Create 55 controlled files and a separate collection for scale testing."""
    if not 200 <= scale <= 300:
        raise ValueError("Scale corpus must contain 200–300 files.")
    if destination.exists():
        raise FileExistsError(f"Choose a new dataset directory: {destination}")
    controlled = destination / "controlled"
    queries = []
    records = {}
    types = [".txt", ".pdf", ".docx"]
    for number, (category, query, stem, text) in enumerate(CASES, start=1):
        extension = (".png" if number % 2 else ".jpg") if category == "ocr" else types[(number - 1) % 3]
        relative = f"{category}/{stem}{extension}"
        write_file(controlled / relative, text)
        records[relative] = text
        queries.append({"id": f"Q{number:02}", "category": category, "query": query, "known_file": relative})
    for number, text in enumerate(DISTRACTORS, start=1):
        relative = f"other/note_{number:02}{types[number % 3]}"
        write_file(controlled / relative, text)
        records[relative] = text
    (destination / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    with (destination / "judgments.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query_id", "file_id", "relevant", "evidence"])
        for query in queries:
            for relative, text in records.items():
                relevant = relative == query["known_file"]
                evidence = f"Filename: {relative}" if relevant and query["category"] == "metadata" else text
                writer.writerow([query["id"], relative, int(relevant),
                                 evidence if relevant else f"Different information need: {text}"])
    import shutil
    large = destination / "scale"
    shutil.copytree(controlled, large)
    for number in range(scale - len(records)):
        # Extra records are not included in the controlled relevance judgments.
        text = f"Archive entry {number + 1}. {DISTRACTORS[number % len(DISTRACTORS)]}"
        write_file(large / f"archive/entry_{number + 1:03}{types[number % 3]}", text)
    print(f"Created {len(records)} controlled and {scale} scale files in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--scale", type=int, default=250)
    args = parser.parse_args()
    generate(args.output, args.scale)
