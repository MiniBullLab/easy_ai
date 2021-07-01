#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


class Point2d():
    """
    Helper class to define 2dBox
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

    def __repr__(self):
        return "[%.3f, %.3f]" % (self.x, self.y)


class Point3d():
    """
    Helper class to define 3dBox
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        return


class MyObject():

    def __init__(self):
        self.name = ""
        self.class_id = -1
        self.difficult = 0


class Rect2D(MyObject):
    """
    class to define 3dBox
    """

    def __init__(self, x_min=0, y_min=0, x_max=0, y_max=0):
        super().__init__()
        self.min_corner = Point2d(x_min, y_min)
        self.max_corner = Point2d(x_max, y_max)

    def copy(self):
        b = Rect2D()
        b.name = self.name
        b.class_id = self.class_id
        b.difficult = self.difficult
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def center(self):
        x = (self.max_corner.x + self.min_corner.x) / 2
        y = (self.max_corner.y + self.min_corner.y) / 2
        return x, y

    def width(self):
        return self.max_corner.x - self.min_corner.x

    def height(self):
        return self.max_corner.y - self.min_corner.y

    def getTuple(self):
        return (self.min_corner.x, self.min_corner.y,
                self.max_corner.x, self.max_corner.y)

    def getVector(self):
        return [self.min_corner.x, self.min_corner.y,
                self.max_corner.x, self.max_corner.y]

    def __repr__(self):
        return '%s:%d %d %d %d' % (self.name, self.min_corner.x, self.min_corner.y, self.max_corner.x, self.max_corner.y)


class KeyPoint2D(Rect2D):

    def __init__(self):
        super().__init__()
        self.key_points = []
        self.key_points_flag = ()
        self.key_points_skeleton = ()

    def copy(self):
        b = KeyPoint2D()
        b.name = self.name
        b.class_id = self.class_id
        b.difficult = self.difficult
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        b.key_points = self.key_points[:]
        b.key_points_flag = self.key_points_flag
        b.key_points_skeleton = self.key_points_skeleton
        return b

    def set_rect2d(self, rect2d):
        self.min_corner = rect2d.min_corner
        self.max_corner = rect2d.max_corner

    def get_rect2d(self):
        b = Rect2D()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def clear_key_points(self):
        self.key_points.clear()
        self.key_points_flag = ()
        self.key_points_skeleton = ()

    def add_key_points(self, point):
        self.key_points.append(point)

    def get_key_points(self):
        return self.key_points

    def set_key_points_flag(self, flag):
        if isinstance(flag, (list, tuple)):
            self.key_points_flag = tuple(flag)
        else:
            self.key_points_flag = (flag,)

    def get_key_points_flag(self):
        return self.key_points_flag

    def set_key_points_skeleton(self, skeleton):
        self.key_points_skeleton = tuple(skeleton)


class Rect3D(MyObject):
    """
    class to define 3dBox
    """

    def __init__(self):
        super().__init__()
        self.center_corner = Point3d(0, 0, 0)
        self.size = Point3d(0, 0, 0)


class OCRObject(Rect2D):

    def __init__(self):
        super().__init__()
        self.polygon = []
        self.object_text = None
        self.language = None
        self.text_confidence = []
        self.object_confidence = 0

    def copy(self):
        b = OCRObject()
        b.name = self.name
        b.class_id = self.class_id
        b.difficult = self.difficult
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        b.polygon = self.polygon[:]
        b.object_text = self.object_text
        b.text_confidence = self.text_confidence[:]
        return b

    def copy_polygon(self, polygon_object):
        self.polygon = polygon_object.polygon[:]
        self.object_confidence = polygon_object.object_confidence
        self.class_id = polygon_object.class_id
        self.name = polygon_object.name

    def set_rect2d(self, rect2d):
        self.min_corner = rect2d.min_corner
        self.max_corner = rect2d.max_corner

    def get_rect2d(self):
        b = Rect2D()
        b.min_corner = self.min_corner
        b.max_corner = self.max_corner
        return b

    def clear_polygon(self):
        self.polygon.clear()

    def add_point(self, point):
        self.polygon.append(point)

    def get_polygon(self):
        return self.polygon

    def set_text(self, temp_str):
        self.object_text = temp_str

    def get_text(self):
        return self.object_text

    def __repr__(self):
        return '{}: {}'.format(self.object_text, self.polygon)


class Polygon2dObject(MyObject):

    def __init__(self):
        super().__init__()
        self.polygon = []
        self.object_confidence = 0

    def copy(self):
        b = OCRObject()
        b.name = self.name
        b.class_id = self.class_id
        b.difficult = self.difficult
        b.polygon = self.polygon[:]
        b.object_confidence = self.object_confidence
        return b

    def clear_polygon(self):
        self.polygon.clear()

    def add_point(self, point):
        self.polygon.append(point)

    def get_polygon(self):
        return self.polygon


class DetectionObject(Rect2D):

    def __init__(self):
        super().__init__()
        self.classIndex = -1
        self.classConfidence = 0
        self.objectConfidence = 0


class DetectionKeyPoint(KeyPoint2D):

    def __init__(self):
        super().__init__()
        self.classIndex = -1
        self.keypoint_confidence = 0
        self.objectConfidence = 0


