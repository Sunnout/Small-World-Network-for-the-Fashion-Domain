from skimage.transform import resize

# WARNING: Estes imports dão erro no Notebook e tem de se tirar o from PWPro
from PWPro import DataManager as dm
from PWPro import FeatureExtractor as fe


class ImageProcessor:

    def __init__(self):
        
        #Guardar info em ficheiro e adicionar flags para buscar ou escrever
        self.colors = []
        self.grads = []

        self.db_imgs = dm.get_img_names()

        for img_name in self.db_imgs:
            img = dm.get_img(img_name)

            img = self.center_crop_image(img, size=224)

            #Extract features
            color_hist, bins = fe.hoc(img)
            grad_hist = fe.my_hog(img) #gray scale?
            
            self.colors.append(color_hist)
            self.grads.append(grad_hist)


    def process_single_img(self, img_name, img):
        """Função para processar imagem de input
        Busca às features das imagens da bd se a imagem pertencer à bd
        Senão calcula as features para a imagem nova"""

        if img_name in self.db_imgs:
            # Get features from DB
            img_color = self.colors.index(img_name)
            img_grad = self.grads.index(img_name)
        else:
            # Extract features
            img = self.center_crop_image(img, size=224)
            img_color, bins = fe.hoc(img)
            img_grad = fe.my_hog(img)  # gray scale?


    @staticmethod
    def center_crop_image(im, size=224):

        if im.shape[2] == 4:  # Remove the alpha channel
            im = im[:, :, 0:3]

        # Resize so smallest dim = 224, preserving aspect ratio
        h, w, _ = im.shape
        if h < w:
            im = resize(image=im, output_shape=(224, int(w * 224 / h)))
        else:
            im = resize(im, (int(h * 224 / w), 224))

        # Center crop to 224x224
        h, w, _ = im.shape
        im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

        return im
