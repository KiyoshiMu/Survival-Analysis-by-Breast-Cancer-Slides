import staintools, cv2, os, sys

def stain_norm(std_img, f_p, dst):

    standardizer = staintools.BrightnessStandardizer()
    i_std = staintools.read_image(std_img)
    stain_normalizer = staintools.StainNormalizer(method='vahadane')
    i_standard = standardizer.transform(i_std)
    stain_normalizer.fit(i_standard)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(f_p):
        img = staintools.read_image(os.path.join(f_p, f))
        i_normalized = stain_normalizer.transform(standardizer.transform(img))
        cv2.imwrite(os.path.join(dst, os.path.basename(f)), i_normalized)

def main():
    std_img = sys.argv[1]
    f_p = sys.argv[2]
    dst = sys.argv[3]
    stain_norm(std_img, f_p, dst)

if __name__ == '__main__':
    main()
        
        