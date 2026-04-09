# def plot_google_map(lat, lon, zoom=11, size=(640, 640), scale=2, maptype="roadmap",
#                     markers=None, api_key="YOUR_API_KEY", return_image=False):
#     import requests
#     import numpy as np
#     from PIL import Image
#     from io import BytesIO

#     url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}"
#     url += f"&size={size[0]}x{size[1]}&scale={scale}&maptype={maptype}&key={api_key}"

#     if markers:
#         for m_lat, m_lon, color in markers:
#             url += f"&markers=color:{color}%7C{m_lat},{m_lon}"

#     response = requests.get(url)
#     if response.status_code == 200:
#         img = Image.open(BytesIO(response.content))
#         deg_per_pix = 156543.03392 * np.cos(np.radians(lat)) / (2**zoom) / 111320
#         width_deg = size[0] * deg_per_pix
#         height_deg = size[1] * deg_per_pix
#         extent = [
#             lon - width_deg / 2,
#             lon + width_deg / 2,
#             lat - height_deg / 2,
#             lat + height_deg / 2
#         ]
#         if return_image:
#             return np.array(img), extent
#         else:
#             return extent
#     return None, None

def plot_google_map(lat, lon, zoom=11, size=(640, 640), scale=2, maptype="roadmap",
                    markers=None, api_key="YOUR_API_KEY", return_image=False):
    import requests
    import numpy as np
    from PIL import Image
    from io import BytesIO

    # Build URL
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}"
    url += f"&size={size[0]}x{size[1]}&scale={scale}&maptype={maptype}&key={api_key}"

    # Add markers to URL
    if markers:
        for m_lat, m_lon, color in markers:
            url += f"&markers=color:{color}%7C{m_lat},{m_lon}"

    # Make the request
    response = requests.get(url)
    if response.status_code == 200:
        # Read image and force RGB mode
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)  # This will be shape (H, W, 3), dtype=uint8 ✅

        # Calculate approximate lat/lon extent
        deg_per_pix = 156543.03392 * np.cos(np.radians(lat)) / (2**zoom) / 111320
        width_deg = size[0] * deg_per_pix
        height_deg = size[1] * deg_per_pix
        extent = [
            lon - width_deg / 2,
            lon + width_deg / 2,
            lat - height_deg / 2,
            lat + height_deg / 2
        ]

        if return_image:
            return img_np, extent
        else:
            return extent
    else:
        print("Failed to load map image.")
        return None, None
