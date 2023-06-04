import numpy as np
import time
import cv2
import os
import flet as ft

def main(page: ft.page):
    page.title = "Yolo"
    page.window_center()
    page.window_height = 750
    page.window_width = 1200

    txt1 = ft.Text(value=("Path to image "), size = 20, text_align = ft.TextAlign.CENTER)
    txt2 = ft.Text(value=("Yolo base path "), size = 20, text_align = ft.TextAlign.CENTER)
    txt3 = ft.Text(value="Confidence ", size = 20, text_align = ft.TextAlign.CENTER)
    txt4 = ft.Text(value="Threshold ", size = 20, text_align = ft.TextAlign.CENTER)

    #get path to image
    def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            "".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )       
        selected_files.update() 

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()
    page.overlay.append(pick_files_dialog)

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()

    # Save file dialog
    def save_file_result(e: ft.FilePickerResultEvent):
        save_file_path.value = e.path if e.path else "Cancelled!"
        save_file_path.update()

    save_file_dialog = ft.FilePicker(on_result=save_file_result)
    save_file_path = ft.Text()
    #get yolo-coco path
    def get_directory_result(e: ft.FilePickerResultEvent):
        directory_path.value = e.path if e.path else "Cancelled!"
        directory_path.update()

    get_directory_dialog = ft.FilePicker(on_result=get_directory_result)
    directory_path = ft.Text()

    # hide all dialogs in overlay
    page.overlay.extend([pick_files_dialog, save_file_dialog, get_directory_dialog])


    gap_slider1 = ft.Slider(
        min=0,
        max=5,
        divisions=5,
        value=0,
        label="{value}",
    )

    gap_slider2 = ft.Slider(
        min=0,
        max=10,
        divisions=10,
        value=0,
        label="{value}",
    )

    def button_clicked(e):
        global image, directory
        page.go("/waiting")
        # load the COCO class labels our YOLO model was trained on
        gap_slider1.value = float(gap_slider1.value/10)
        gap_slider2.value = float(gap_slider2.value/10)
        print(str(selected_files)[16:-2])
        print(str(directory_path).strip("text {'value':'} '"))

        labelsPath = os.path.sep.join([str(directory_path).strip("text {'value':'} '"), "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([str(directory_path).strip("text {'value':'} '"), "yolov3.weights"])
        configPath = os.path.sep.join([str(directory_path).strip("text {'value':'} '"), "yolov3.cfg"])

        # load YOLO object detector trained on COCO dataset
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # load input image and grab its spatial dimensions
        image = cv2.imread(str(save_file_path)[16:-2])

        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > gap_slider1.value:
                    # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually  returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, gap_slider1.value,
            gap_slider2.value)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        directory = os.path.dirname(os.path.abspath(__file__))     
        cv2.imwrite(directory+'/out.png', image)
        directory = directory.replace("\\", "/")
        directory += "/out.png"

        page.go("/result")

    def yes_click(e):
            page.window_destroy()

    #page route
    def route_change(route):
        page.views.clear()
        page.views.append(
            ft.View(
                "/",
                [
                    ft.Row(
                        [
                            txt1, 
                            ft.ElevatedButton(
                                "Pick the image",
                                on_click=lambda _:save_file_dialog.save_file(),
                            ),
                            save_file_path
                        ],
                    ),
                    ft.Row(
                        [
                            txt2, 
                            ft.ElevatedButton(
                                "Pick",
                                on_click=lambda _:get_directory_dialog.get_directory_path()
                            ),
                            directory_path,
                        ]
                    ),
                    ft.Row(
                        [
                            txt3, 
                            gap_slider1,
                        ]
                    ),
                    ft.Row(
                        [
                            txt4, 
                            gap_slider2,
                        ],
                    ),
                    
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "OK", 
                                on_click=button_clicked
                            ),
                            
                        ]
                    )    
                    
                ],
                scroll = ft.ScrollMode.HIDDEN,
                padding=ft.padding.only(left = 250, top = 200)
            ),
        )
        #next page
        if page.route == "/waiting":
            page.views.append(
                ft.View(
                    "/waiting",
                    [   
                        ft.ProgressRing(),
                    ],
                    scroll = ft.ScrollMode.HIDDEN,
                    padding=ft.padding.only(left = 600, top = 350)
                )
            )
        if page.route == "/result":
            page.views.append(
                ft.View(
                    "/result",
                    [   

                        ft.Text(f"Stroed in {directory}", size = 20, text_align = ft.TextAlign.CENTER),
                        ft.Image(src=f"{directory}"),
                        ft.ElevatedButton("Leave", on_click=yes_click),
                    ],
                    scroll = ft.ScrollMode.HIDDEN,
                    horizontal_alignment = ft.CrossAxisAlignment.CENTER,
                    vertical_alignment = ft.MainAxisAlignment.CENTER,
                )
            )
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)

ft.app(target = main)

