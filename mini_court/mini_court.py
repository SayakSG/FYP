import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt:
    def __init__(self, frame):
        print("[MiniCourt] Initializing...")
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        print("[MiniCourt] Initialization complete.")

    def convert_meters_to_pixels(self, meters):
        pixels = convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )
        print(f"[MiniCourt] {meters} meters -> {pixels} pixels")
        return pixels

    def set_court_drawing_key_points(self):
        print("[MiniCourt] Setting court key points...")
        self.drawing_key_points = [0] * 28
        # TODO: Setup actual mini-court keypoints here.
        print(f"[MiniCourt] Court key points set.")

    def set_court_lines(self):
        print("[MiniCourt] Setting court lines...")
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),
            (0, 1), (8, 9), (10, 11), (10, 11), (2, 3)
        ]
        print(f"[MiniCourt] Court lines set.")

    def set_mini_court_position(self):
        print("[MiniCourt] Setting mini court position...")
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        print(f"[MiniCourt] Court: ({self.court_start_x}, {self.court_start_y}) to ({self.court_end_x}, {self.court_end_y})")

    def set_canvas_background_box_position(self, frame):
        print("[MiniCourt] Setting background box...")
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        print(f"[MiniCourt] Background box: ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")

    def draw_court(self, frame):
        print("[MiniCourt] Drawing court...")
        for i in range(0, len(self.drawing_key_points), 2):
            x, y = int(self.drawing_key_points[i]), int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for line in self.lines:
            start = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start, end, (0, 0, 0), 2)

        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        cv2.line(frame, (self.drawing_key_points[0], net_y), (self.drawing_key_points[2], net_y), (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        print("[MiniCourt] Drawing background...")
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        print("[MiniCourt] Drawing mini court on frames...")
        output_frames = []
        for idx, frame in enumerate(frames):
            print(f"[MiniCourt] Frame {idx}")
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index, player_height_pixels, player_height_meters):
        distance_x_pixels, distance_y_pixels = measure_xy_distance(object_position, closest_key_point)

        distance_x_meters = convert_pixel_distance_to_meters(
            distance_x_pixels, player_height_meters, player_height_pixels)
        distance_y_meters = convert_pixel_distance_to_meters(
            distance_y_pixels, player_height_meters, player_height_pixels)

        mini_x = self.convert_meters_to_pixels(distance_x_meters)
        mini_y = self.convert_meters_to_pixels(distance_y_meters)

        mini_key_x = self.drawing_key_points[closest_key_point_index * 2]
        mini_key_y = self.drawing_key_points[closest_key_point_index * 2 + 1]

        mini_court_position = (mini_key_x + mini_x, mini_key_y + mini_y)
        print(f"[MiniCourt] Mapped position: {mini_court_position}")
        return mini_court_position

def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
    print("[MiniCourt] Converting bounding boxes...")
    output_player_boxes = []
    output_ball_boxes = []

    DEFAULT_PLAYER_HEIGHT = 1.75  # meters
    player_heights = {
        1: constants.PLAYER_1_HEIGHT_METERS,
        2: constants.PLAYER_2_HEIGHT_METERS
    }

    for frame_num, player_bbox in enumerate(player_boxes):
        print(f"[MiniCourt] Frame {frame_num}")
        ball_box = ball_boxes[frame_num].get(1)
        ball_position = get_center_of_bbox(ball_box)

        output_player_bboxes = {}

        # Handle players
        for player_id in [1, 2]:
            bbox = player_bbox.get(player_id)
            if bbox is not None:
                foot = get_foot_position(bbox)
                keypoint_idx = get_closest_keypoint_index(foot, original_court_key_points, [0, 2, 12, 13])
                keypoint = (original_court_key_points[keypoint_idx * 2], original_court_key_points[keypoint_idx * 2 + 1])

                frame_window = range(max(0, frame_num - 20), min(len(player_boxes), frame_num + 50))
                heights = [
                    get_height_of_bbox(player_boxes[i].get(player_id, bbox))
                    for i in frame_window if player_id in player_boxes[i]
                ]
                max_height = max(heights) if heights else get_height_of_bbox(bbox)

                player_height = player_heights.get(player_id, DEFAULT_PLAYER_HEIGHT)
                mini_position = self.get_mini_court_coordinates(foot, keypoint, keypoint_idx, max_height, player_height)
                output_player_bboxes[player_id] = mini_position
            else:
                # If player missing, set to (0, 0)
                output_player_bboxes[player_id] = (0, 0)

        output_player_boxes.append(output_player_bboxes)

        # Handle ball
        if ball_position is not None:
            closest_player_id = min(output_player_bboxes.keys(), key=lambda pid: measure_distance(ball_position, output_player_bboxes[pid]))
            ball_mini_position = output_player_bboxes.get(closest_player_id, (0, 0))
        else:
            ball_mini_position = (0, 0)

        output_ball_boxes.append({1: ball_mini_position})

    print("[MiniCourt] Bounding box conversion done.")
    return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        print("[MiniCourt] Drawing points...")
        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions):
                for _, position in positions[frame_num].items():
                    x, y = position
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        return frames

    def get_width_of_mini_court(self):
        return self.court_drawing_width
