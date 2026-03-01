import ezdxf
from ezdxf.addons import r12writer
import numpy as np

class DXFConverter:
    def __init__(self):
        pass

    def convert(self, lines, circles, output_file):
        doc = ezdxf.new("R2000")
        doc.units = ezdxf.units.MM
        dxf = doc.modelspace()

        for line in lines:
            dxf.add_line((line[0][0], line[0][1]), (line[0][2], line[0][3]))

        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            dxf.add_circle((circle[0], circle[1]), circle[2])

        doc.saveas(output_file)