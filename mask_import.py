"""functions for laser mask creation, version 1.2"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from PIL import Image, ImageDraw

def create_movexy_2dlist(
        bounds_x: list,     # [min_x, max_x] x boundaries of the tissue scan    float / int
        bounds_y: list,     # [min_y, max_y] y boundaries of the tissue scan    float / int
        subgrp_w: int,      # number of subgroups, left-right                   int             >=1
        subgrp_h: int,      # number of subgroups, top-bottom                   int             >=1
        subgrp_g: float,    # distance between adjacent subgroups               float / int     >=0
        region_n: int,      # number of FOV scans, on one side of a subgroup    int             >=1
                            # (i.e. a 4x4 subgroup should have region_n = 4)
        region_s: float,    # xy size of FOV scans in given units               float / int     > 0
        region_d: float,    # distance between adjacent FOV scans               float / int     >=0
        enable_f: bool,     # show a pyplot preview for FOV scans               bool
        enable_r: bool,     # rounds all xy values if set to true               bool
):
    """
    Return the xy delta values of a scan scheme in a 2D list.
    
    x : [min_x, max_x] x boundaries of the tissue scan.
    y : [min_y, max_y] y boundaries of the tissue scan.
    w : number of subgroups, left-right.
    h : number of subgroups, top-bottom.
    g : gap size between adjacent subgroups.
    n : number of FOV scans, on one side of a subgroup.
    s : xy size of FOV scans in given units.
    d : gap size between adjacent FOV scans.
    f : show a pyplot preview for FOV scans.
    r : rounds all xy values if set to true.
    """
    # ==================================== variable processing ====================================
    # declare variables to match each parameter, localize any modifications
    # doing so allows LabVIEW to pass parameter values instead of to modify
    # declare variables that depends on enable_r
    if enable_r:
        x = round(0.5*(bounds_x[0] + bounds_x[1]))    # center x coordinate
        y = round(0.5*(bounds_y[0] + bounds_y[1]))    # center x coordinate
        g = round(subgrp_g)
        d = round(region_d)
        s = round(region_s)
    else:
        x = 0.5*(bounds_x[0] + bounds_x[1])   # center x coordinate
        y = 0.5*(bounds_y[0] + bounds_y[1])   # center x coordinate
        g = subgrp_g
        d = region_d
        s = region_s
    # declare variables not affected by enable_r
    w = subgrp_w
    h = subgrp_h
    n = region_n
    f = enable_f
    rtn = []        # return list
    tmp = []        # list to store centers of FOV scans
    txt = ''        # string to store parameter previews
    # append values to txt, construct an fstring saving important parameters to preview on the side
    txt += f'Scan Config: ({n}x{n})*({w}x{h})\n'
    txt += f'Number of FOV Scans: {n*n*w*h}\n\n'
    txt += f'Scan Range x: [{x-0.5*((s+d)*n-d)*w-g*(w-1)},{x+0.5*((s+d)*n-d)*w+g*(w-1)}]\n'
    txt += f'Scan Range y: [{y-0.5*((s+d)*n-d)*h-g*(h-1)},{y+0.5*((s+d)*n-d)*h+g*(h-1)}]\n\n'
    txt += f'Tissue Range x: [{bounds_x[0]},{bounds_x[1]}]\n'
    txt += f'Tissue Range y: [{bounds_y[0]},{bounds_y[1]}]\n\n'
    txt += f'Size of FOV Scans: {s}\n'
    txt += f'Gap Sizes between Subgroups: {g}\n'
    txt += f'Gap Sizes between FOV Scans: {d}'
    # =============================== append FOV center coordinates ===============================
    # create subgroups and save all FOV coordinates into tmp
    tmp = create_scheme_2dlist(x, y, w, h, g, n, s, d)
    # calculate the xy changes (xy deltas) for the first FOV
    rtn.append([tmp[0][0] - x, tmp[0][1] - y])
    # loop and calculate the xy changes for the rest of FOVs
    for i in range(1, len(tmp)):
        # append xy delta values into rtn
        rtn.append([(tmp[i][0] - tmp[i - 1][0]),
                    (tmp[i][1] - tmp[i - 1][1])])
        if f:   # if preview_f is enabled
            # save FOV previews to pyplot
            create_region_pyplot(tmp[i][0], tmp[i][1], s, s, 'r', 'g', 'left', 'top', i + 1)
    if f:   # if preview_f is enabled
        # save the center of the scan scheme into pyplot
        create_region_pyplot(
            x,
            y,
            bounds_x[1] - bounds_x[0],
            bounds_y[1] - bounds_y[0],
            'b',
            'b',
            'right',
            'bottom',
            f'({x},{y})'
        )
        # create a region in pyplot to preview for the first FOV
        create_region_pyplot(tmp[0][0], tmp[0][1], s, s, 'r', 'g', 'left', 'top', 1)
        # list imporant parameters on the side in a separate box
        plt.text(
            1.05,
            1,
            txt,
            transform = plt.gca().transAxes,
            fontsize = 10,
            verticalalignment='top',
            bbox = dict(facecolor='none', alpha=0.15)
        )
        # set aspect to equal, show preview in a separate window
        plt.gca().set_aspect('equal')
        plt.gcf().set_figwidth(15)
        plt.gcf().set_figheight(7.5)
        plt.tight_layout()
        plt.show()
    return rtn


def create_scheme_2dlist(
        center_x: float,    # center x coordinate                               float / int
        center_y: float,    # center y coordinate                               float / int
        subgrp_w: int,      # number of subgroups, left-right                   int             >=1
        subgrp_h: int,      # number of subgroups, top-bottom                   int             >=1
        subgrp_g: float,    # distance between adjacent subgroups               float / int     >=0
        region_n: int,      # number of FOV scans, on one side of a subgroup    int             >=1
                            # (i.e. a 4x4 subgroup should have region_n = 4)
        region_s: float,    # xy size of FOV scans in given units               float / int     > 0
        region_d: float,    # distance between adjacent FOV scans               float / int     >=0
):
    """
    Return a 2D list of xy coordinates for all FOVs in the scan scheme (in subgroup order).
    
    x : center x coordinate.
    y : center y coordinate.
    w : number of subgroups, left-right.
    h : number of subgroups, top-bottom.
    g : distance between adjacent subgroups.
    n : number of FOV scans, on one side of a subgroup.
    s : xy size of FOV scans in given units.
    d : distance between adjacent FOV scans.
    """
    # ============================== initialize and adjust variables ==============================
    x = center_x                    # cursor x position, initially at the center
    y = center_y                    # cursor y position, initially at the center
    w = subgrp_w                    # number of subgroups, left-right
    h = subgrp_h                    # number of subgroups, top-bottom
    g = subgrp_g                    # distance between adjacent subgroups
    n = region_n                    # number of FOV scans, on one side of a subgroup
    s = region_s                    # xy size of FOV scans in given units
    d = region_d                    # distance between adjacent FOV scans
    r = (s + d) * n + g - d         # distance between centers of adjacent subgroups
    rtn = []                        # return list
    tmp = []                        # list for centers of subgroups
    grp = []                        # list for centers of FOVs in one subgroup
    # ============================== append coordinates by subgroups ==============================
    # find the center coordinate for the first subgroup (top-left)
    x -= ((w - 1) / 2.0) * r
    y += ((h - 1) / 2.0) * r
    # append center coordinates of all subgroups to tmp
    for i in range (0, h):
        for j in range (0, w):
            tmp.append([(x + j * r), (y - i * r)])
    # loop through tmp and append coordinates of all FOVs into rtn
    for subgrp in tmp:
        # append coordinates of all FOVs in that subgroup into grp
        grp = create_subgrp_2dlist(subgrp[0], subgrp[1], n, s, d)
        # append coordinates of FOVs stored in grp into rtn
        for l in grp:
            rtn.append(l)
    # return rtn once the loop ends
    return rtn


def create_subgrp_2dlist(
        cursor_x: float,    # cursor x coordinate for this subgroup             float / int
        cursor_y: float,    # cursor y coordinate for this subgroup             float / int
        region_n: int,      # number of FOV scans, on one side of a subgroup    int             >=1
                            # (i.e. a 4x4 subgroup should have region_n = 4)
        region_s: float,    # xy size of FOV scans in given units               float / int     > 0
        region_d: float     # distance between adjacent FOV scans               float / int     >=0
):
    """
    Return a 2D list of xy coordinates for all FOVs in one subgroup (in spiral order).

    x : center x coordinate.
    y : center y coordinate.
    n : number of FOV scans, on one side of a subgroup.
    s : xy size of FOV scans in given units.
    d : distance between adjacent FOV scans.
    """
    # ============================== initialize and adjust variables ==============================
    r = region_s + region_d     # distance between the centers of adjacent FOVs
    c = "down"                  # cardinal directions for the cursor to move to
    x = cursor_x                # cursor x position, initially at the center of the subgroup
    y = cursor_y                # cursor y position, initially at the center of the subgroup
    i = 0                       # index of the FOV that the cursor is currently at
    j = 0                       # number of times the append direction was changed
    l = 0                       # number of times the cursor moves before it turns
    rtn = []                    # return list
    # ===================== append cursor's coordinates as it spirals outward =====================
    # if region_n is even, adjust cursor xy to the center of adjacent FOV at the lower-right corner
    if (region_n % 2) == 0:
        x += (0.5 * r)
        y -= (0.5 * r)
    # spiral counter-clockwise outward, loop until all center coordinates for FOVs are appended
    while i < (region_n * region_n):
        # for every 2 changes in the appending direction, l += 1
        if (j % 2) == 0:
            l += 1
        # turn the appending direction counter-clockwise
        if c == "left":
            c = "up"
        elif c == "up":
            c = "right"
        elif c == "right":
            c = "down"
        elif c == "down":
            c = "left"
        # move the cursor as it appends its location for l times
        for _ in range(0, l):
            # increment the number of FOVs appended
            i += 1
            # append the current cursor coordinates as a 1D list
            rtn.append([x, y])
            if c == "left":     # move one FOV left
                x -= r
            elif c == "up":     # move one FOV up
                y += r
            elif c == "right":  # move one FOV right
                x += r
            elif c == "down":   # move one FOV down
                y -= r
        # increment the number of turns
        j += 1
    # return rtn once the loop ends
    return rtn


def create_region_pyplot(
        x: float,       # center x coordinate                       float / int
        y: float,       # center y coordinate                       float / int
        w: float,       # size of FOV over x axis                   float / int
        h: float,       # size of FOV over y axis                   float / int
        c = 'b',        # color to be used to plot the center       str / chr
        e = 'b',        # color to be used to plot the border       str / chr
        f = 'left',     # alignment of the center i to x axis       str / chr
        v = 'top',      # alignment of the center i to y axis       str / chr
        i = "",         # value to be displayed at the center       (printable)
        j = "",         # image to be displayed at the center       (file path)
        a = 1           # alpha value of all marking elements       float / int     0-1
):
    """
    Store a rectangle with width = w and height = h at (x,y), marked with i.
    
    x : center x coordinate.
    y : center y coordinate.
    w : size of FOV over x axis.
    h : size of FOV over y axis.
    c : color to be used to plot the center.
    e : color to be used to plot the border.
    f : alignment of the center i to x axis.
    v : alignment of the center i to y axis.
    i : value to be displayed at the center.
    j : image to be displayed at the center.
    a : alpha value of all marking elements
    """
    # declare two lists to store corner coordinates
    corner_x = []
    corner_y = []
    # bottom left (start)
    corner_x.append(x - 0.5*w)
    corner_y.append(y - 0.5*h)
    # top left
    corner_x.append(x - 0.5*w)
    corner_y.append(y + 0.5*h)
    # top right
    corner_x.append(x + 0.5*w)
    corner_y.append(y + 0.5*h)
    # bottom right
    corner_x.append(x + 0.5*w)
    corner_y.append(y - 0.5*h)
    # bottom left (finish)
    corner_x.append(x - 0.5*w)
    corner_y.append(y - 0.5*h)
    # plot i as label
    plt.plot(x, y, 'o', color = c, alpha = a)
    plt.text(x, y, i, ha = f, va = v, alpha = a)
    # plot j as image and rectX - rectY as lines
    if j != "":
        # open image with PIL
        img = Image.open(j)
        # store img, stretch its dimension to fit the current FOV
        ax = plt.gca()
        ax.imshow(np.fliplr(np.flipud(img)), extent=(x + 0.5*w,x - 0.5*w,y + 0.5*h,y - 0.5*h))
        # invert the axes back as imshow will invert x and y axis
        ax.invert_xaxis()
        ax.invert_yaxis()
        # plot rectX - recty with linestyle = ':'
        plt.plot(corner_x, corner_y, ':', color = e, alpha = a)
    else:
        # plot rectX - recty with linestyle = '-'
        plt.plot(corner_x, corner_y, '-', color = e, alpha = a)


def create_cpmask_global(
        xylist_l: list,     # FOV xy delta values stored as 2D lists        float / int
        images_f: str,      # file path to the original image folder        (file path)
        region_s: float,    # xy size of FOV scans in given units           float / int     > 0
        subgrp_w: int,      # number of subgroups, left-right               int             >=1
        subgrp_h: int,      # number of subgroups, top-bottom               int             >=1
):
    """
    Create a global laser mask combining all images from the designated folder.

    l : FOV xy delta values stored as 2D lists.
    f : file path to the original image folder.
    s : xy size of FOV scans in given units.
    w : number of subgroups, left-right.
    h : number of subgroups, top-bottom.
    g : save ImageJ txt files for masks.
    """
    # ==================================== access stored masks ====================================
    files = []
    for file in os.scandir(images_f):
        if file.is_file():
            if os.path.basename(file)[-13:] == '_cp_masks.png':
                files.append(file.path)
    if len(files) != len(xylist_l):
        raise IndexError("Number of xy coordinate pairs != Number of png masks.")
    if len(files) == 0:
        raise IndexError("No image ending with \'_cp_masks.png\' found in the assigned directory.")
    # ================================= assemble individual masks =================================
    # initialize a new mask as the global mask
    h, w, _ = io.imread(files[0]).shape
    if h != w:
        raise ValueError("Individual laser mask must be a perfect square.")
    glb = Image.new('1', [w * subgrp_w, h * subgrp_h], 255)
    # append existing masks to the global mask
    x = int(w * subgrp_w / 2 - w / 2)
    y = int(h * subgrp_h / 2 - h / 2)
    for i, mask in enumerate(files):
        x += int(xylist_l[i][0] / region_s * w)
        y -= int(xylist_l[i][1] / region_s * h)
        glb.paste(Image.open(mask), [x, y])
        # os.remove(mask)
    glb.save(images_f + '/_global.png')


def create_cpmask_pngtxt(
        images_f: str,          # file path to the original image folder     (file path)
        gettxt_g: bool,         # save ImageJ txt files for masks            bool
        sample_d = None         # average (pixel) diameter for all cells     'None' / int
):
    """
    Create laser masks for all multichannel images stored in a folder using cellpose 2.

    f : file path to the original image folder.
    g : save ImageJ txt files for masks.
    d : average (pixel) diameter for all cells.
    """
    # ==================================== variable processing ====================================
    io.logger_setup()
    # model_type = 'cyto'/'nuclei'/'cyto2'/'cyto3'
    model = models.Cellpose(model_type='cyto3')
    # define which channels to run segmentation on
    # segmentation channels = [cytoplasm, nucleus]
    # includes: 0 = grayscale, 1 = R, 2 = G, 3 = B
    channels = [0,0]
    # ======================================= mask creation =======================================
    # loops for all files in the image folder
    # make sure there are no other file types
    files = []
    for file in os.scandir(images_f):
        if file.is_file():
            files.append(file.path)
    for filename in files:
        # ----------------------------------- cp initialization -----------------------------------
        # run cp2 for every image, save mask image as png, and export coordinates for mask outlines
        img = io.imread(filename)
        masks, flows, _styles, _diams = model.eval(img, diameter=sample_d, channels=channels)
        io.save_masks(img, masks, flows, filename, png=True, tif=False, save_txt=True)
        # read coordinates for mask outlines from txt
        list_x = []
        list_y = []
        txt = filename[:-4] + '_cp_outlines.txt'
        with open(txt, encoding="utf-8") as text:
            content = text.readlines()
            for line in content:
                region = line[:(len(line)-1)].split(",")
                newline_x = []
                newline_y = []
                odd = True
                for cord in region:
                    if odd:
                        newline_x.append(int(cord))
                        odd = False
                    else:
                        newline_y.append(int(cord))
                        odd = True
                list_x.append(newline_x)
                list_y.append(newline_y)
            text.close()
        # ---------------------------------- mask reconstruction ----------------------------------
        # create a new mask from coordinates for mask outlines stored in txt
        msk = Image.new('1',Image.open(filename).size, 0)
        pxl_msk = msk.load()
        for i, cell in enumerate(list_x):
            for j, pixels in enumerate(cell):
                # color in all peripheral pixels as white
                pxl_msk[pixels, list_y[i][j]] = 255
        # find a reference point (stored as blk) outside all of the cells in the png mask by cp2
        ref = filename[:-4] + '_cp_masks.png'
        pxl_ref = Image.open(ref)
        for i in range(pxl_ref.size[0]):
            for j in range(pxl_ref.size[1]):
                if pxl_ref.load()[i,j] == 0:
                    blk = [i,j]
                    # color in any pixels surrounded by mask outlines but aren't inside the mask
                    if pxl_msk[i,j] == 0:
                        pxl_msk[i,j] = 255
        # apply white floodfill for the rest of the image
        ImageDraw.floodfill(msk, blk, 255)
        # remove temporary files, save the new mask image
        if not gettxt_g:
            os.remove(txt)
        os.remove(ref)
        msk.save(ref)
