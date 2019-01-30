import os, sys, openslide
"""Extract tiff image under 20X power. However, the file is too large to be analyzed further. It's  useless now."""
def get_files(dir_p):
    return [os.path.join(item[0], fn) for item in os.walk(dir_p) 
    for fn in item[2] if item[2] and fn[-4:]=='.svs']

def get_name(p):
    return os.path.splitext(os.path.basename(p))[0]

def extract(slide_p, dst):
    img = openslide.OpenSlide(slide_p)
    power = openslide.OpenSlide(slide_p).properties.get('openslide.objective-power', False)
    if power:
        level = 0
        size = img.level_dimensions[0]
        out = img.read_region((0, 0), level, size)

        if power != '20':
            ratio = int(int(power) / 20)
            new_size = (int(size[0]/ratio), int(size[1]/ratio))
            out = out.resize(new_size)

        out.save(os.path.join(dst, get_name(slide_p)+'.tif'))

if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True)
    extract(sys.argv[1], sys.argv[2])