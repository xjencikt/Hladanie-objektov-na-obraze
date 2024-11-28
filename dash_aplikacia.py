import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import cv2
import base64
import numpy as np

img = cv2.imread("S1019L01.jpg")

img_gray = cv2.imread("S1019L01.jpg", cv2.IMREAD_GRAYSCALE)

circles = [
    {"name": "spodna_cast", "center": (145, 3), "radius": 255, "color": (0, 0, 255)},
    {"name": "horna_cast", "center": (140, 326), "radius": 260, "color": (0, 255, 0)},
    {"name": "duhovka", "center": (160, 160), "radius": 107, "color": (255, 0, 0)},
    {"name": "zrenicka", "center": (158, 166), "radius": 38, "color": (255, 255, 0)}
]

for circle in circles:
    cv2.circle(img, circle["center"], circle["radius"], circle["color"], 2)

_, img_encoded = cv2.imencode('.png', img)
img_base64 = base64.b64encode(img_encoded).decode()

ret, buffer = cv2.imencode('.png', img_gray)
encoded_image_gray = base64.b64encode(buffer).decode('utf-8')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Circle Detection App", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.H3("Processed Image", style={'textAlign': 'center'}),
            dcc.Slider(
                id='param1-slider',
                min=1,
                max=300,
                step=1,
                value=75,
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={'always_visible': True},
                included=False
            ),
            html.Div(id='param1-text'),
            dcc.Slider(
                id='param2-slider',
                min=1,
                max=300,
                step=1,
                value=40,
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={'always_visible': True},
                included=False
            ),
            html.Div(id='param2-text'),
            dcc.Slider(
                id='param3-slider',
                min=1,
                max=300,
                step=1,
                value=40,
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={'always_visible': True},
                included=False
            ),
            html.Div(id='param3-text'),
            dcc.Slider(
                id='param4-slider',
                min=1,
                max=300,
                step=1,
                value=40,
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={'always_visible': True},
                included=False
            ),
            html.Div(id='param4-text'),
            dcc.Slider(
                id='param5-slider',
                min=1,
                max=300,
                step=1,
                value=40,
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={'always_visible': True},
                included=False
            ),
            html.Div(id='param5-text'),

        ], style={'width': '48%', 'margin': '0 auto', 'padding-bottom': '60px', 'font-size': '20px',
                  'text-align': 'center'}
        ),
    ]),

    html.Div(id='output-image-container', style={'padding-bottom': '60px', 'font-size': '20px', 'line-height': '1.5'}),
    html.Div([
        dcc.RadioItems(
            id='image-selection',
            options=[
                {'label': 'Original Image', 'value': 'original'},
                {'label': 'Histogram Equalization', 'value': 'he'},
                {'label': 'CLAHE', 'value': 'clahe'},
                {'label': 'Gaussian Noise', 'value': 'gaussian'},
                {'label': 'Non-local Means', 'value': 'nlm'},
                {'label': 'Canny Edge Detection', 'value': 'canny'}
            ],
            value='original',
            labelStyle={'display': 'inline-block', 'margin-right': '20px', 'padding-left': '60px',
                        'padding-right': '60px', \
                        'padding-bottom': '20px'}
        ),
        html.Div(id='image-container')
    ]),
    html.Div([
        html.Label('Clip Limit for CLAHE:'),
        dcc.Slider(
            id='clip-limit-slider',
            min=1,
            max=10,
            step=0.1,
            value=2.0,
            marks={i: str(i) for i in range(1, 11)}
        ),
        html.Div(id='clip-limit-output')
    ]),
    html.Div([
        html.Label('Histogram Equalization Strength:'),
        dcc.Slider(
            id='hist-equalization-slider',
            min=1,
            max=10,
            step=0.1,
            value=1.0,
            marks={i: str(i) for i in range(1, 11)}
        ),
        html.Div(id='hist-equalization-output')
    ]),
    html.Div([
        html.Label('Gaussian Noise Sigma:'),
        dcc.Slider(
            id='gaussian-sigma-slider',
            min=0,
            max=101,
            step=10,
            value=0,
            marks={i: str(i) for i in range(0, 101, 10)}
        ),
        html.Div(id='gaussian-sigma-output')
    ]),
    html.Div([
        html.Label('Non-local Means H Value:'),
        dcc.Slider(
            id='nlm-h-slider',
            min=1,
            max=101,
            step=10,
            value=10,
            marks={i: str(i) for i in range(1, 101, 10)}
        ),
        html.Div(id='nlm-h-output')
    ]),
    html.Div([
        html.Label('Canny Max Threshold:'),
        dcc.Slider(
            id='canny-max-threshold-slider',
            min=0,
            max=241,
            step=1,
            value=214,
            marks={i: str(i) for i in range(1, 241, 40)}
        ),
        html.Div(id='canny-max-threshold-output')
    ]),
    html.Div([
        html.Label('Canny Ratio Pre-Min Threshold:'),
        dcc.Slider(
            id='canny-ratio-slider',
            min=0.1,
            max=0.9,
            step=0.2,
            value=0.5,
            marks={i: str(i) for i in range(0, 101, 10)}
        ),
        html.Div(id='canny-ratio-output')
    ]),
    html.Div([
        html.Div([
            html.H3("Original Image"),
            html.Img(id='original-image', src='data:image/png;base64,{}'.format(encoded_image_gray))
        ], style={'width': 'auto', 'float': 'left', 'margin-left': '400px', 'font-size': '20px',
                  'text-align': 'center'}),
        html.Div([
            html.H3("Image with Circles"),
            html.Img(src='data:image/png;base64,{}'.format(img_base64))
        ], style={'width': 'auto', 'float': 'right', 'margin-right': '400px', 'font-size': '20px',
                  'text-align': 'center'})
    ], style={'width': '100%', 'overflow': 'hidden'})

])


@app.callback(
    [Output('param1-text', 'children'),
     Output('param2-text', 'children'),
     Output('param3-text', 'children'),
     Output('param4-text', 'children'),
     Output('param5-text', 'children'),
     Output('output-image-container', 'children'),
     Output('image-container', 'children')],
    [Input('image-selection', 'value'),
     Input('clip-limit-slider', 'value'),
     Input('hist-equalization-slider', 'value'),
     Input('gaussian-sigma-slider', 'value'),
     Input('nlm-h-slider', 'value'),
     Input('canny-max-threshold-slider', 'value'),
     Input('canny-ratio-slider', 'value'),
     Input('param1-slider', 'value'),
     Input('param2-slider', 'value'),
     Input('param3-slider', 'value'),
     Input('param4-slider', 'value'),
     Input('param5-slider', 'value')]
)
def update_image_and_output(selected_image, clip_limit, hist_equalization, gaussian_sigma, nlm_h, canny_max_threshold,
                            canny_ratio, param1, param2, param3, param4, param5):
    img_gray = cv2.imread("S1019L01.jpg", cv2.IMREAD_GRAYSCALE)

    ret, buffer = cv2.imencode('.png', img_gray)
    encoded_image_gray = base64.b64encode(buffer)

    param_texts = [
        f"Accumulator resolution: {param1}  ",
        f" Limit for votes in accumulator cell: {param2}",
        f" Value of the upper limit in Canny detector: {param3}",
        f" Min. center distance: {param4}",
        f" Min. circle size: {param5}"
    ]

    if selected_image == 'he':
        img_gray = cv2.equalizeHist(img_gray)
        img_gray = cv2.convertScaleAbs(img_gray, alpha=hist_equalization)
    elif selected_image == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)
    elif selected_image == 'gaussian':
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), gaussian_sigma)
    elif selected_image == 'nlm':
        img_gray = cv2.fastNlMeansDenoising(img_gray, None, h=nlm_h)
    elif selected_image == 'canny':
        img_gray = cv2.Canny(img_gray, canny_max_threshold, canny_ratio * canny_max_threshold)

    image = img_gray.copy()
    dst = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=param1, minDist=param4,
                               param1=param2, param2=param3, minRadius=param5, maxRadius=0)

    output_text = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if 55 > r > 0:
                print("X:", x)
                print("Y:", y)
                print("R:", r)
                output_text = "zrenicka"
                circle_color = (0, 0, 255)
                cv2.circle(image, (x, y), r, circle_color, 2)
            elif 110 > r > 80:
                print("X:", x)
                print("Y:", y)
                print("R:", r)
                output_text = "duhovka"
                circle_color = (0, 255, 0)
                cv2.circle(image, (x, y), r, circle_color, 2)
            elif y - r + 200 < 0 and r - 350 < x < r - 150:
                print("X:", x)
                print("Y:", y)
                print("R:", r)
                output_text = "dolna cast"
                circle_color = (255, 0, 0)
                cv2.circle(image, (x, y), r, circle_color, 2)
            elif r + 600 > y and r - 150 < x < r - 50:
                print("X:", x)
                print("Y:", y)
                print("R:", r)
                output_text = "horna cast"
                circle_color = (255, 0, 0)
                cv2.circle(image, (x, y), r, circle_color, 2)

    retval, buffer = cv2.imencode('.png', image)
    img_str = 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')

    output_image = html.Div([
        html.Span(param_texts[0], style={'margin-right': '30px', 'margin-left': '90px'}),
        html.Span(param_texts[1], style={'margin-right': '30px'}),
        html.Span(param_texts[2], style={'margin-right': '30px'}),
        html.Span(param_texts[3], style={'margin-right': '30px'}),
        html.Span(param_texts[4]),
        html.Div(style={'margin-left': 'auto'}, children=[
            html.Img(src=img_str, style={'display': 'block', 'margin': 'auto', 'padding': '20px'})
        ])
    ])

    return param_texts[0], param_texts[1], param_texts[2], param_texts[3], param_texts[4], output_image, output_text


if __name__ == '__main__':
    app.run_server(debug=True)
