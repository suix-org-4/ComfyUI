import os
import shutil
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

import onnxruntime as ort

import insightface
from insightface.app.common import Face
import torch

import folder_paths
import comfy.model_management as model_management
from modules.shared import state

from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
    progress_bar,
    progress_bar_reset
)
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")
hyperswap_path = os.path.join(models_path, "hyperswap")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)

        model_filename = os.path.basename(model_path)
        if "hyperswap" in model_filename.lower():
            model_path = os.path.join(folder_paths.models_dir, "hyperswap", model_filename)
            FS_MODEL = ort.InferenceSession(model_path, providers=providers)
        elif "reswapper" in model_filename.lower():
            model_path = os.path.join(folder_paths.models_dir, "reswapper", model_filename)
            FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
        else:
            FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


# Функция для получения 5 ключевых точек из объекта Face
def get_landmarks_5(face):
    if hasattr(face, 'landmark_5') and face.landmark_5 is not None:
        return face.landmark_5
    elif hasattr(face, 'kps') and face.kps is not None:
        return face.kps
    elif hasattr(face, 'landmark') and face.landmark is not None:
        if face.landmark.shape[0] >= 68:
            idxs = [36, 45, 30, 48, 54]
            return face.landmark[idxs]
    return None

# Функция для вычисления аффинного преобразования
def get_affine_transform(src_pts, dst_pts):
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M

# Создаём градиентную маску овальной формы без обрезки 
def create_gradient_mask(crop_size=256):
    # 1. Создаём пустую маску (все пиксели = 0)
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    
    # 2. Определяем центр и размеры эллипса
    center = (crop_size // 2, crop_size // 2)
    axes = (int(crop_size * 0.35), int(crop_size * 0.4))
    
    # 3. Рисуем эллипс (заполняем белым цветом, значение=1.0)
    cv2.ellipse(
        mask,          # Массив для рисования
        center,        # Центр эллипса
        axes,          # Полуоси (ширина, высота)
        angle=0,       # Угол поворота
        startAngle=0,  # Начальный угол дуги
        endAngle=360,  # Конечный угол дуги (360 = полный эллипс)
        color=1.0,     # Значение для заполнения (белый = 1.0)
        thickness=-1   # -1 = заполнить всю область эллипса   
    )
    
    # 4. Применяем размытие для плавных краёв
    blur_ksize = 15  # Нечётное число, чтобы ядро было симметричным
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    
    # 5. Ограничим значения в диапазоне [0, 1]
    mask = np.clip(mask, 0, 1)
    
    return mask

def paste_back(target_img, swapped_face, M, crop_size=256):
    
    # 1. Создание мягкой маски (Эрозия + Размытие)
    mask = create_gradient_mask(crop_size)

    # Преобразуем в трехканальную маску
    mask_3c = np.stack([mask] * 3, axis=2)

    # 2. Получаем размеры целевого изображения
    h, w = target_img.shape[:2]

    # 3. Обратное преобразование (WARP_INVERSE_MAP) для лица И маски
    # Для лица (INTER_LANCZOS4 — высококачественная интерполяция)
    inv_face = cv2.warpAffine(
        swapped_face.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # Для маски (INTER_CUBIC — плавные границы)
    inv_mask = cv2.warpAffine(
        mask_3c,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # 4. Ограничение значений маски [0, 1]
    inv_mask = np.clip(inv_mask, 0, 1)

    # 5. Дополнительное размытие для устранения артефактов
    inv_mask = cv2.GaussianBlur(inv_mask, (3, 3), 0)

    # 6. Плавное наложение
    target_img_float = target_img.astype(np.float32)
    inv_face_float = inv_face.astype(np.float32)
    result = target_img_float * (1.0 - inv_mask) + inv_face_float * inv_mask

    # 7. Ограничение результата [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def visualize_points(img, points, color=(0, 255, 0)):
    img = img.copy()
    for p in points:
        cv2.circle(img, tuple(p.astype(int)), 3, color, -1)

# Итоговая функция run_hyperswap с аффинным преобразованием
def run_hyperswap(session, source_face, target_face, target_img):
    # 1. Подготовка эмбеддинга
    source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

    # 2. Получаем 5 точек target
    target_landmarks_5 = get_landmarks_5(target_face)
    visualize_points(target_img, target_landmarks_5, (0, 255, 0))  # Зеленые точки
    
    if target_landmarks_5 is None:
        return None, None

    # 3. Определение эталонных точек для выравнивания 256x256 (FFHQ Alignment)
    std_landmarks_256 = np.array([
        [ 84.87, 105.94],  # Левый глаз
        [171.13, 105.94],  # Правый глаз
        [128.00, 146.66],  # Кончик носа
        [ 96.95, 188.64],  # Левый уголок рта
        [159.05, 188.64]   # Правый уголок рта
    ], dtype=np.float32)

    # Вычисляем аффинную матрицу
    M = get_affine_transform(target_landmarks_5.astype(np.float32), std_landmarks_256)
    
    # Применяем аффинное преобразование с новой матрицей M
    crop = cv2.warpAffine(target_img, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # 4. Преобразуем crop для модели
    crop_input = crop[:, :, ::-1].astype(np.float32) / 255.0  # RGB -> [0,1]
    crop_input = (crop_input - 0.5) / 0.5  # Нормализация
    crop_input = crop_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # 5. Инференс
    try:
        output = session.run(None, {'source': source_embedding, 'target': crop_input})[0][0]
    except:
        return target_img

    # 6. Обратная нормализация
    output = (output * 0.5 + 0.5) * 255.0  # [-1..1] -> [0..255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    output = output[:, :, ::-1]  # BGR -> RGB
    
    return output, M # Возвращаем лицо (256x256) и матрицу M


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
    face,
    face_index,
    gender_condition,
    operated: str,
    order: str,
):
    filtered_faces = [
        f for f in face
        if (gender_condition == 0) or
        (gender_condition == 1 and f.sex == "F") or
        (gender_condition == 2 and f.sex == "M")
    ]

    gender = "Female" if gender_condition == 1 else "Male" if gender_condition == 0 else ""

    if len(filtered_faces) == 0:
        if gender_condition != 0:
            logger.status(f"No faces found for -{gender}-")
        return None, 0, None  # treat as "wrong gender" to skip

    faces_sorted = sort_by_order(filtered_faces, order)

    if face_index >= len(faces_sorted):
        logger.info("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(faces_sorted))
        return None, 0, None

    face_selected = faces_sorted[face_index]

    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, face_selected.sex)

    expected_gender = "F" if gender_condition == 1 else "M"
    if gender_condition != 0 and face_selected.sex != expected_gender:
        logger.info(f"{operated} Face {face_index}: WRONG gender ({face_selected.sex})")
        return face_selected, 1, face_index  # <-- есть, но не тот пол

    return face_selected, 0, face_index

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)

    faces = []
    try:
        faces = face_analyser.get(img_data)
    except:
        logger.error("No faces found")

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0, face_index
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0, None


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img
    bbox = []
    swapped_indexes = []

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image, bbox, swapped_indexes

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)
                
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender, target_face_index = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if "hyperswap" in model:
                                swapped_face_256, M = run_hyperswap(face_swapper, source_face, target_face, result)
                                if swapped_face_256 is not None:
                                    result = paste_back(result, swapped_face_256, M, crop_size=256)
                            elif face_boost_enabled:
                                logger.status(f"Face Boost is enabled (inswapper/reswapper only)")
                                bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                M *= scale
                                result = swapper.in_swap(result, bgr_fake, M)
                            else:
                                result = face_swapper.get(result, target_face, source_face)
                            bbox = [tuple(map(float, target_face.bbox))]
                            swapped_indexes = [target_face_index]
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.info(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_image, bbox, swapped_indexes

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    result_images = target_imgs
    bbox = []
    swapped_indexes = []

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_faces = []
            pbar = progress_bar(len(target_imgs))

            if len(TARGET_IMAGE_LIST_HASH) > 0:
                logger.status(f"Using Hashed Target Face(s) Model...")
            else:
                logger.status(f"Analyzing Target Image...")
            
            for i, target_img in enumerate(target_imgs):
                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    break
                
                target_image_md5hash = get_image_md5hash(target_img)
                if len(TARGET_IMAGE_LIST_HASH) == 0:
                    TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
                    target_image_same = False
                elif len(TARGET_IMAGE_LIST_HASH) == i:
                    TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
                    target_image_same = False
                else:
                    target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
                    if not target_image_same:
                        TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash
                
                logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
                logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

                if len(TARGET_FACES_LIST) == 0:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST = [target_face]
                elif len(TARGET_FACES_LIST) == i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST.append(target_face)
                elif len(TARGET_FACES_LIST) != i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST[i] = target_face
                elif target_image_same:
                    # logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
                    target_face = TARGET_FACES_LIST[i]
                

                # logger.status(f"Analyzing Target Image {i}...")
                # target_face = analyze_faces(target_img)
                if target_face is not None:
                    target_faces.append(target_face)
                
                pbar.update(1)

            progress_bar_reset(pbar)
            
            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images, bbox, swapped_indexes

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                results = target_imgs
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)

                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                pbar = progress_bar(len(target_imgs))

                logger.status(f"Swapping...")
                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        # Reading results to make current face swap on a previous face result
                        # logger.status(f"Swapping...")
                        for i, (target_img, target_face) in enumerate(zip(results, target_faces)):
                            target_face_single, wrong_gender, target_face_index = get_face_single(target_img, target_face, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                            if target_face_single is not None and wrong_gender == 0:
                                result = target_img
                                if "hyperswap" in model:
                                    swapped_face_256, M = run_hyperswap(face_swapper, source_face, target_face_single, result)
                                    if swapped_face_256 is not None:
                                        result = paste_back(result, swapped_face_256, M, crop_size=256)
                                elif face_boost_enabled:
                                    logger.status(f"Face Boost is enabled (inswapper/reswapper only)")
                                    bgr_fake, M = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
                                    bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                    M *= scale
                                    result = swapper.in_swap(target_img, bgr_fake, M)
                                else:
                                    result = face_swapper.get(target_img, target_face_single, source_face)
                                results[i] = result
                                bbox.append(tuple(map(float, target_face_single.bbox)))
                                swapped_indexes.append(target_face_index)
                                pbar.update(1)
                            elif wrong_gender == 1:
                                wrong_gender = 0
                                logger.status("Wrong target gender detected")
                                pbar.update(1)
                                continue
                            else:
                                logger.info(f"{i}: No target face found for {face_num}")
                                pbar.update(1)
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                progress_bar_reset(pbar)

                result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_images, bbox, swapped_indexes
