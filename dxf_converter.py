import ezdxf
from ezdxf.addons import r12writer

class DXFConverter:
    def __init__(self):
        pass

    def convert(self, lines, circles, output_file):
        doc = ezdxf.new("R2000")
        doc.units = ezdxf.units.MM
        dxf = doc.modelspace()

        dxf.add_line((0, 0), (17, 23))
        dxf.add_circle((0, 0), radius=2)
        dxf.add_arc((0, 0), radius=3, start_angle=0, end_angle=175)

        doc.saveas(output_file)