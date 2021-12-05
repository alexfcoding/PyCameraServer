# Rendering states dictionary
states_dict = {
    'view_source': False,
    'source_image': "",
    'source_url': "",
    'source_mode': "",
    'output_file_page': "",
    'screenshot_path': "",
    'need_to_create_screenshot': False,
    'screenshot_ready': False,
    'working_on': True,
    'frame_processed': 0,
    'total_frames': 0,
    'options': "",
    'screenshot_lock': False,
    'video_reset_lock': False,
    'video_stop_lock': False,
    'mode_reset_lock': False,
    'source_lock': False,
    'render_mode': "",
    'superres_model': "LAPSRN",
    'esrgan_model': "FALCOON"
}

# Rendering modes dictionary
render_modes_dict = {
    'canny_people_on_background': 'd',
    'canny_people_on_black': 'c',
    'extract_and_replace_background': 'i',
    'extract_and_cut_background': 'g',
    'color_canny': 'j',
    'color_canny_on_background': 'h',
    'color_objects_on_gray_blur': 'l',
    'color_objects_blur': 'm',
    'color_objects_on_gray': 'k',
    'caffe_colorization': 'f',
    'cartoon_effect': 'e',
    'extract_objects_yolo_mode': 'a',
    'text_render_yolo': 'b',
    'denoise_and_sharpen': 'o',
    'sobel': 'p',
    'ascii_painter': 'q',
    'pencil_drawer': 'r',
    'two_colored': 's',
    'upscale_opencv': 'n',
    'upscale_esrgan': 't',
    'boost_fps_dain': 'z'
}

# Default rendering settings
# Values will change with AJAX requests
settings_ajax = {
    'viewSource': False,
    'cannyBlurSliderValue': 5,
    'cannyThresSliderValue': 50,
    'cannyThresSliderValue2': 50,
    'cannyThres2': 50,
    'saturationSliderValue': 100,
    'contrastSliderValue': 100,
    'brightnessSliderValue': 0,
    'positionSliderValue': 1,
    'confidenceSliderValue': 20,
    'lineThicknessSliderValue': 2,
    'denoiseSliderValue': 10,
    'denoiseSliderValue2': 10,
    'sharpenSliderValue': 5,
    'sharpenSliderValue2': 5,
    'rcnnSizeSliderValue': 10,
    'rcnnBlurSliderValue': 17,
    'sobelSliderValue': 3,
    'asciiSizeSliderValue': 4,
    'asciiIntervalSliderValue': 10,
    'asciiThicknessSliderValue': 1,
    'resizeSliderValue': 2,
    'colorCountSliderValue': 32,
    'mode': 'a',
    'superresModel': 'LapSRN',
    'esrganModel': 'FALCOON',
    'urlSource': 'default'
}