"""Generate data/sample/sample.pdf for testing and demos.

Creates a 3-page PDF about the Solar System with enough text to produce
several chunks. Run this once:  python scripts/generate_sample_pdf.py
"""

import os
import sys

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample.pdf")


def main() -> None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print("reportlab is required: pip install reportlab")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    doc = SimpleDocTemplate(OUTPUT, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    pages = [
        {
            "title": "The Solar System: An Overview",
            "body": (
                "The Solar System consists of the Sun and all the objects that orbit it, "
                "including eight planets, dwarf planets, moons, asteroids, and comets. "
                "It formed approximately 4.6 billion years ago from the gravitational collapse "
                "of a giant molecular cloud. The Sun accounts for more than 99.8 percent of "
                "the Solar System's total mass. The four inner planets — Mercury, Venus, "
                "Earth, and Mars — are terrestrial planets, composed primarily of rock and metal. "
                "The four outer planets — Jupiter, Saturn, Uranus, and Neptune — are much larger "
                "and are composed mostly of hydrogen and helium (Jupiter and Saturn) or of ices "
                "alongside hydrogen and helium (Uranus and Neptune, often called ice giants). "
                "Earth is the only known planet to harbour life. It lies in the habitable zone "
                "of the Sun, where liquid water can exist on the surface. The Moon, Earth's only "
                "natural satellite, stabilises Earth's axial tilt and drives ocean tides."
            ),
        },
        {
            "title": "The Inner Planets",
            "body": (
                "Mercury is the smallest planet and the closest to the Sun. Its surface "
                "temperature swings dramatically, from 430 degrees Celsius during the day to "
                "minus 180 degrees Celsius at night, because it has virtually no atmosphere "
                "to retain heat. A Mercurian year lasts only 88 Earth days. "
                "Venus is the second planet and the hottest in the Solar System, with an average "
                "surface temperature of 465 degrees Celsius sustained by a thick carbon dioxide "
                "atmosphere and a runaway greenhouse effect. It rotates in the opposite direction "
                "to most planets, meaning the Sun rises in the west on Venus. "
                "Earth orbits the Sun at an average distance of about 150 million kilometres — "
                "a distance defined as one Astronomical Unit (AU). Its atmosphere, composed "
                "mainly of nitrogen and oxygen, protects life from harmful solar radiation "
                "and regulates temperature. Earth's magnetic field deflects the solar wind. "
                "Mars, known as the Red Planet due to iron oxide on its surface, has the largest "
                "volcano in the Solar System — Olympus Mons, standing 21 kilometres high. "
                "Mars has two small moons, Phobos and Deimos, and evidence suggests liquid "
                "water once flowed across its surface."
            ),
        },
        {
            "title": "The Outer Planets and Beyond",
            "body": (
                "Jupiter is the largest planet, with a mass more than twice that of all other "
                "planets combined. Its Great Red Spot is a storm that has raged for at least "
                "350 years. Jupiter has at least 95 known moons; its four largest — Io, Europa, "
                "Ganymede, and Callisto — were discovered by Galileo in 1610. Europa is of "
                "particular scientific interest because a subsurface ocean of liquid water may "
                "exist beneath its icy crust, making it a candidate for extraterrestrial life. "
                "Saturn is famous for its spectacular ring system, composed of ice and rock "
                "particles ranging from tiny grains to boulders several metres across. Saturn's "
                "moon Titan is the only moon in the Solar System with a dense atmosphere and "
                "surface lakes of liquid methane and ethane. "
                "Uranus and Neptune are ice giants with interiors rich in water, ammonia, and "
                "methane ices. Uranus rotates on its side, with an axial tilt of 98 degrees. "
                "Neptune has the strongest winds in the Solar System, reaching 2,100 kilometres "
                "per hour. Beyond Neptune lies the Kuiper Belt, a region of icy bodies including "
                "the dwarf planet Pluto, which was reclassified from a full planet in 2006. "
                "The Oort Cloud, a vast spherical shell at the far edge of the Solar System, "
                "is the source of long-period comets. The Solar System's influence extends to "
                "the heliopause, where the solar wind meets interstellar space — a boundary "
                "that Voyager 1 crossed in 2012, becoming the first human-made object to "
                "enter interstellar space."
            ),
        },
    ]

    for page in pages:
        story.append(Paragraph(page["title"], styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(page["body"], styles["BodyText"]))
        story.append(Spacer(1, 0.5 * inch))

    doc.build(story)
    print(f"Generated: {os.path.abspath(OUTPUT)}")


if __name__ == "__main__":
    main()
