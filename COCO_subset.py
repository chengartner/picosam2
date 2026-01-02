import os
import shutil
import json
from pycocotools.coco import COCO

# =======================
# USER CONFIGURATION
# =======================
COCO_ROOT = './dataset'  # Root folder of COCO dataset
OUTPUT_ROOT = './subset_COCO'  # Output folder for filtered dataset

TRAIN_IMAGES_DIR = os.path.join(COCO_ROOT, 'train2017')
VAL_IMAGES_DIR = os.path.join(COCO_ROOT, 'val2017')

TRAIN_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', 'instances_train2017.json')
VAL_ANN_FILE = os.path.join(COCO_ROOT, 'annotations', 'instances_val2017.json')

OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_ROOT, 'train2017')
OUTPUT_VAL_DIR = os.path.join(OUTPUT_ROOT, 'val2017')
OUTPUT_ANN_DIR = os.path.join(OUTPUT_ROOT, 'annotations')

INDOOR_SUPERCATEGORIES = {'furniture', 'appliance', 'electronic'}
PERSON_SUPERCATEGORY = 'person'

os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANN_DIR, exist_ok=True)


# =======================
# HELPER FUNCTION
# =======================
def filter_coco_dataset(coco_ann_file, images_dir, output_images_dir, output_ann_file):
    print(f"\nProcessing {coco_ann_file}...")

    coco = COCO(coco_ann_file)

    # --- Get category IDs ---
    cats = coco.loadCats(coco.getCatIds())
    person_cat_ids = [c['id'] for c in cats if c['supercategory'] == PERSON_SUPERCATEGORY]
    indoor_cat_ids = [c['id'] for c in cats if c['supercategory'] in INDOOR_SUPERCATEGORIES]

    print("Person category IDs:", person_cat_ids)
    print("Indoor category IDs:", indoor_cat_ids)

    # --- Get image IDs containing person ---
    person_img_ids = set(coco.getImgIds(catIds=person_cat_ids))

    # --- Get image IDs containing indoor objects ---
    indoor_img_ids = set()
    for cat_id in indoor_cat_ids:
        indoor_img_ids.update(coco.getImgIds(catIds=[cat_id]))

    print(f"Found {len(person_img_ids)} images with person.")
    print(f"Found {len(indoor_img_ids)} images with indoor objects.")

    # --- Filtered image IDs (person OR indoor) ---
    filtered_img_ids = person_img_ids.union(indoor_img_ids)
    print(f"Total images after union: {len(filtered_img_ids)}")

    # --- Copy images to new folder ---
    for img_id in filtered_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        src = os.path.join(images_dir, filename)
        dst = os.path.join(output_images_dir, filename)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # --- Filter annotations ---
    ann_ids = coco.getAnnIds(imgIds=list(filtered_img_ids))
    annotations = coco.loadAnns(ann_ids)

    # --- Filter categories to only those used ---
    used_cat_ids = {ann['category_id'] for ann in annotations}
    categories = coco.loadCats(list(used_cat_ids))

    # --- Filtered images info ---
    images = coco.loadImgs(list(filtered_img_ids))

    # --- Save new COCO annotation JSON ---
    filtered_coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_ann_file, 'w') as f:
        json.dump(filtered_coco, f)

    print(f"Saved filtered annotations to {output_ann_file}")


# =======================
# PROCESS TRAIN AND VAL
# =======================
filter_coco_dataset(
    coco_ann_file=TRAIN_ANN_FILE,
    images_dir=TRAIN_IMAGES_DIR,
    output_images_dir=OUTPUT_TRAIN_DIR,
    output_ann_file=os.path.join(OUTPUT_ANN_DIR, 'instances_filtered_train2017.json')
)

filter_coco_dataset(
    coco_ann_file=VAL_ANN_FILE,
    images_dir=VAL_IMAGES_DIR,
    output_images_dir=OUTPUT_VAL_DIR,
    output_ann_file=os.path.join(OUTPUT_ANN_DIR, 'instances_filtered_val2017.json')
)

print("\nCOCO filtering complete!")