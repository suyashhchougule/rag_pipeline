import glob
import os

import cv2
import torch
from PIL import Image
from typing import Optional
from transformers import AutoProcessor, VisionEncoderDecoderModel
from utils.utils import *

class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)
        
        Args:
            prompt: Text prompt or list of prompts to guide the model
            image: PIL Image or list of PIL Images to process
            
        Returns:
            Generated text or list of texts from the model
        """
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        # Prepare image
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values.half().to(self.device)
        
        # Prepare prompt
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt"
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0
        )
        
        # Process output
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        # Clean prompt text from output
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
            
        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results


def process_document(document_path, model, save_dir, max_batch_size=None):
    """Parse documents with two stages - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()
    
    if file_ext == '.pdf':
        # Process PDF file
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")
        
        all_results = []
        
        # Process each page
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # Process this page (don't save individual page results)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)
        
        # Save combined results for multi-page PDF
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        
        return combined_json_path, all_results
    
    else:
        # Process regular image file
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True):
    """Process a single image (either from file or converted from PDF page)
    
    Args:
        image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output file
        max_batch_size: Maximum batch size for processing
        save_individual: Whether to save individual results (False for PDF pages)
        
    Returns:
        Tuple of (json_path, recognition_results)
    """
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)

    # Stage 2: Element-level content parsing
    padded_image, dims = prepare_image(image)
    recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        # Create a dummy image path for save_outputs function
        dummy_image_path = f"{image_name}.jpg"  # Extension doesn't matter, only basename is used
        json_path = save_outputs(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    # Store text and table elements separately
    text_elements = []  # Text elements
    table_elements = []  # Table elements
    figure_results = []  # Image elements (no processing needed)
    previous_box = None
    reading_order = 0

    # Collect elements to process and group by type
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    
                    # For figure regions, store relative path instead of base64
                    figure_results.append(
                        {
                            "label": label,
                            "text": f"![Figure](figures/{figure_filename})",
                            "figure_path": f"figures/{figure_filename}",
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # Prepare element for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    
                    # Group by type
                    if label == "tab":
                        table_elements.append(element_info)
                    else:  # Text elements
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Initialize results list
    recognition_results = figure_results.copy()
    
    # Process text elements (in batches)
    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)
    
    # Process table elements (in batches)
    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []
    
    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)
        
        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)
        
        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
    
    return results


class DolphinDocumentParser:
    def __init__(self, model_path: str, save_dir: Optional[str] = None, max_batch_size: int = 16):
        self.model = DOLPHIN(model_path)
        self.save_dir = save_dir
        self.max_batch_size = max_batch_size

    def parse_documents(self, input_path):
        # Collect document files
        if os.path.isdir(input_path):
            file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
            document_files = []
            for ext in file_extensions:
                document_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            document_files = sorted(document_files)
        else:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path {input_path} does not exist")
            file_ext = os.path.splitext(input_path)[1].lower()
            supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
            if file_ext not in supported_exts:
                raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported_exts}")
            document_files = [input_path]

        save_dir = self.save_dir or (
            input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        )
        setup_output_dirs(save_dir)

        results = []
        for file_path in document_files:
            # Determine expected output JSON path
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            file_ext = os.path.splitext(file_path)[1].lower()
            mdsave_dir = os.path.join(save_dir, "markdown")
            expected_md = os.path.join(mdsave_dir, f"{base_name}.md")
            print(expected_md)
            if os.path.exists(expected_md):
                print(f"Skipping {file_path}: already processed.")
                continue
            try:
                print(f"Processing {file_path}")
                json_path, recognition_results = process_document(
                    document_path=file_path,
                    model=self.model,
                    save_dir=save_dir,
                    max_batch_size=self.max_batch_size,
                )
                results.append((file_path, json_path, recognition_results))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        return results




# model_path = "/home/user/rag_pipeline/Dolphin/hf_model/"         # Path to your HuggingFace model or model ID
# save_dir = "./output"             # Directory to save results (optional)
# max_batch_size = 256               # Batch size for processing (optional)

# parser = DolphinDocumentParser(model_path, save_dir, max_batch_size)


# input_path = "./docs"  # Folder or single file to process

# results = parser.parse_documents(input_path)