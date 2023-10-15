import numpy as np

class Cube():
    def __init__(self, id: int = None, center: (int, int) = None, polygon: [int] = None, team: int = None) -> None:
        self.id = id
        self.center = center
        self.polygon = polygon
        self.team = team
        self.orientation: ((int, int), (int, int)) = None

    def setId(self, id: int) -> None:
        self.id = id
        
    def setCenter(self, center: (int, int)) -> None:
        self.center = center
    
    def setPolygon(self, polygon: [int]) -> None:
        self.polygon = polygon

    def setTeam(self, team: int) -> None:
        self.team = team
    
    def calculateOrientation(self) -> None:
        "calculat the two possible orientations of the cube based on the polygon of the mask"
        #TODO
        pass

    def to_str(self) -> str:
        string: str = str(self.id) + " " + str(self.center) + " " + str(self.team) + " " + str(self.polygon)
        return string
        


def cubes_from_results(results) -> [Cube]:
    detections: [Cube] = []

    classes: [int] = []
    centers: [(int, int)] = []
    masks: [[int]] = []

    for result in results:
        for box in result.boxes: 
            cls = box.cls.cpu().numpy()
            classes.append(cls[0])
           
            coordinates = box.xyxy.cpu().numpy()
            coordinates = coordinates[0]
            center_x = (coordinates[0] + coordinates[2])/2
            center_y = (coordinates[1] + coordinates[3])/2
            centers.append((center_x, center_y))
        for mask in result.masks:
            masks.append((np.int32([mask.xy]))[0])


    for index, iterator in enumerate(classes):
        cube: Cube = Cube(index, centers[index], masks[index], classes[index])
        detections.append(cube)
    
    return detections


    
