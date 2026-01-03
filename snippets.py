# Add sliders for contrast control
fig.update_layout(
    title=f"Band {images[i].band} - Adjust sliders to control contrast",
    width=900,
    height=900,
    xaxis=dict(title="X (pixels)", constrain='domain'),
    yaxis=dict(title="Y (pixels)", scaleanchor='x', constrain='domain'),
    sliders=[
        dict(
            active=0,
            y=-0.02,
            yanchor="top",
            currentvalue=dict(prefix="Min: ", font=dict(size=12)),
            pad=dict(t=50, b=10),
            steps=[
                dict(
                    args=[{"zmin": np.percentile(img_data, pct)}],
                    label=f"{pct}%",
                    method="restyle"
                ) for pct in [0, 0.1, 0.5, 1, 2, 5, 10, 20]
            ]
        ),
        dict(
            active=2,
            y=-0.12,
            yanchor="top",
            currentvalue=dict(prefix="Max: ", font=dict(size=12)),
            pad=dict(t=50, b=10),
            steps=[
                dict(
                    args=[{"zmax": np.percentile(img_data, pct)}],
                    label=f"{pct}%",
                    method="restyle"
                ) for pct in [90, 95, 98, 99, 99.5, 99.9, 99.99, 100]
            ]
        )
    ]
)



data = {'x': [], 'y': [], 'r_half_pix': [], 'flux': [], 'area': [], 'band': []}
for image in images:
    data['x'].append(image.catalog.xcentroid[0])
    data['y'].append(image.catalog.ycentroid[0])
    # estimate half-light radius in pixels
    data['r_half_pix'].append(image.catalog.fluxfrac_radius(0.5)[0])
    data['flux'].append(image.catalog.segment_flux[0],)
    data['area'].append(image.catalog.area[0])
    data['band'].append(image.band)

df = pd.DataFrame(data)
display(df)

