class RandAugment:
    def __init__(self, n=9, m=0.5):
        self.n = n
        self.m = m  # [0, 30] in paper, but we use [0, 1] for simplicity
        self.augment_list = [
            self.auto_contrast, self.equalize, self.rotate, self.solarize, 
            self.color, self.contrast, self.brightness, self.sharpness,
            self.shear_x, self.shear_y, self.translate_x, self.translate_y,
            self.posterize, self.solarize_add, self.invert, self.identity
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img

    def auto_contrast(self, img):
        return ImageOps.autocontrast(img)

    def equalize(self, img):
        return ImageOps.equalize(img)

    def rotate(self, img):
        return TF.rotate(img, self.m * 30)

    def solarize(self, img):
        return TF.solarize(img, int((1 - self.m) * 255))

    def color(self, img):
        return TF.adjust_saturation(img, 1 + self.m)

    def contrast(self, img):
        return TF.adjust_contrast(img, 1 + self.m)

    def brightness(self, img):
        return TF.adjust_brightness(img, 1 + self.m)

    def sharpness(self, img):
        return ImageEnhance.Sharpness(img).enhance(1 + self.m)

    def shear_x(self, img):
        return TF.affine(img, 0, [0, 0], 1, [self.m, 0])

    def shear_y(self, img):
        return TF.affine(img, 0, [0, 0], 1, [0, self.m])

    def translate_x(self, img):
        return TF.affine(img, 0, [int(self.m * img.size[0] / 3), 0], 1, [0, 0])

    def translate_y(self, img):
        return TF.affine(img, 0, [0, int(self.m * img.size[1] / 3)], 1, [0, 0])

    def posterize(self, img):
        return TF.posterize(img, int((1 - self.m) * 8))

    def solarize_add(self, img):
        return TF.solarize(TF.adjust_brightness(img, 1 + self.m), int((1 - self.m) * 255))

    def invert(self, img):
        return TF.invert(img) if random.random() < 0.5 else img

    def identity(self, img):
        return img

class Mixup(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, batch):
        images, labels = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam

class CutMix(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, batch):
        images, labels = batch
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, H, W = images.shape
        cx = np.random.uniform(0, W)
        cy = np.random.uniform(0, H)
        w = W * np.sqrt(1 - lam)
        h = H * np.sqrt(1 - lam)
        x0 = int(np.clip(cx - w // 2, 0, W))
        y0 = int(np.clip(cy - h // 2, 0, H))
        x1 = int(np.clip(cx + w // 2, 0, W))
        y1 = int(np.clip(cy + h // 2, 0, H))
        index = torch.randperm(batch_size)
        images[:, :, y0:y1, x0:x1] = images[index, :, y0:y1, x0:x1]
        lam = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

class RandomErasing(nn.Module):
    def __init__(self, probability=0.25, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        super().__init__()
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def forward(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                else:
                    img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                return img
        return img
