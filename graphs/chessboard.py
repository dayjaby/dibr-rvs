from PIL import Image, ImageDraw
import numpy as np

tbl = np.array(
[[0,0,0,0,0,0,0,0],
 [4,1,4,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,4,1,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0],
 [0,0,0,0,0,4,4,1],]
)

colorTable = {
 0: 'gray',
 1: 'brown',
 2: 'blue',
 3: 'white',
 4: 'green'
}

size = 24
pad = 2

def draw_board(tbl,output,xlabel=None):
    image = Image.new("RGB", ((tbl.shape[1]+2)*size, (tbl.shape[0]+2)*size), 'white')
    draw_square = ImageDraw.Draw(image).rectangle
    draw_text = ImageDraw.Draw(image).text

    if tbl.shape[1]>1:
        for x in xrange(tbl.shape[1]):
            if xlabel is None:
                draw_text([(x+1)*size+4,4],str(x+1),'black')
            else:
                draw_text([(x+1)*size+4,4],xlabel[x],'black')
    if tbl.shape[0]>1:
        for y in xrange(tbl.shape[0]):
            draw_text([4,(y+1)*size+4],str(y+1),'black')

    for y in xrange(tbl.shape[0]):
        for x in xrange(tbl.shape[1]):
            draw_square([(x+1)*size,(y+1)*size,(x+2)*size-pad,(y+2)*size-pad], fill=colorTable[tbl[y,x]])
    image.save(output)

draw_board(tbl,"chessboard.png")
tbl = np.array([[0,0,2,0,0,2,0,0,2,0,2,0,2,0,0,0]])
draw_board(tbl,"extrapolation_proof1.png")
tbl = np.array([[0,0,1,0,0,3,0,0,1,0,3,0,1,0,0,0]])
xlabel = ["1","2","3","4","5","","6","7","8","9","","10","11","12","13","14"]
draw_board(tbl,"extrapolation_proof2.png",xlabel=xlabel)
