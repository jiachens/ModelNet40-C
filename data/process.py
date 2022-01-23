import os

SHAPE = ["airplane",
"bathtub",
"bed",
"bench",
"bookshelf",
"bottle",
"bowl",
"car",
"chair",
"cone",
"cup",
"curtain",
"desk",
"door",
"dresser",
"flower_pot",
"glass_box",
"guitar",
"keyboard",
"lamp",
"laptop",
"mantel",
"monitor",
"night_stand",
"person",
"piano",
"plant",
"radio",
"range_hood",
"sink",
"sofa",
"stairs",
"stool",
"table",
"tent",
"toilet",
"tv_stand",
"vase",
"wardrobe",
"xbox"
]

if __name__ == '__main__':
    for object in SHAPE:
        g = os.walk("data/ModelNet40/"+object+"/test")
        for path,dir_list,file_list in g:  
            for file in file_list:
                # print(file)
                with open(os.path.join(path,file), "r") as f:
                    lines = f.readlines()
                if len(lines[0]) == 4:
                    continue
                else:
                    lines.insert(0,'OFF\n')
                    lines[1] = lines[1][3:]
                    # print(lines)
                    with open(os.path.join(path,file), "w") as f:
                        for line in lines:
                            f.write(line)
