import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance,
)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiniCourt:
    def __init__(self, frame: np.ndarray):
        logger.info("[MiniCourt] Initializing...")

        self.drawing_rectangle_width: int = 250
        self.drawing_rectangle_height: int = 500
        self.buffer: int = 50
        self.padding_court: int = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        self.width = 680  # or however you set it
        self.height = 300

        logger.info("[MiniCourt] Initialization complete.")

    def get_width_of_mini_court(self):
        return self.width

    def set_canvas_background_box_position(self, frame: np.ndarray):
        logger.debug("[MiniCourt] Setting background box...")
        h, w = frame.shape[:2]
        self.end_x: int = w - self.buffer
        self.end_y: int = self.buffer + self.drawing_rectangle_height
        self.start_x: int = self.end_x - self.drawing_rectangle_width
        self.start_y: int = self.end_y - self.drawing_rectangle_height
        logger.debug(
            f"[MiniCourt] Background box: ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})"
        )

    def set_mini_court_position(self):
        logger.debug("[MiniCourt] Setting mini court position...")
        self.court_start_x: int = self.start_x + self.padding_court
        self.court_start_y: int = self.start_y + self.padding_court
        self.court_end_x: int = self.end_x - self.padding_court
        self.court_end_y: int = self.end_y - self.padding_court
        self.court_drawing_width: int = self.court_end_x - self.court_start_x

    def set_court_drawing_key_points(self):
        logger.debug("[MiniCourt] Setting court key points...")
        self.drawing_key_points: List[int] = [
            0
        ] * 28  # Set actual key points appropriately

    def set_court_lines(self):
        logger.debug("[MiniCourt] Setting court lines...")
        self.lines: List[Tuple[int, int]] = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3),
        ]

    def convert_meters_to_pixels(self, meters: float) -> int:
        pixels = convert_meters_to_pixel_distance(
            meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width
        )
        return pixels

    def draw_background_rectangle(self, frame: np.ndarray) -> np.ndarray:
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED,
        )
        out = frame.copy()
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, 0.5, shapes, 0.5, 0)[mask]
        return out

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        for i in range(0, len(self.drawing_key_points), 2):
            x, y = int(self.drawing_key_points[i]), int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for start_idx, end_idx in self.lines:
            start = (
                int(self.drawing_key_points[start_idx * 2]),
                int(self.drawing_key_points[start_idx * 2 + 1]),
            )
            end = (
                int(self.drawing_key_points[end_idx * 2]),
                int(self.drawing_key_points[end_idx * 2 + 1]),
            )
            cv2.line(frame, start, end, (0, 0, 0), 2)

        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        cv2.line(
            frame,
            (self.drawing_key_points[0], net_y),
            (self.drawing_key_points[2], net_y),
            (255, 0, 0),
            2,
        )

        return frame

    def draw_mini_court(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        logger.info("[MiniCourt] Drawing mini court on frames...")
        return [
            self.draw_court(self.draw_background_rectangle(frame.copy()))
            for frame in frames
        ]

    def get_mini_court_coordinates(
        self,
        object_position: Tuple[int, int],
        closest_key_point: Tuple[int, int],
        closest_key_point_index: int,
        player_height_pixels: float,
        player_height_meters: float,
    ) -> Tuple[int, int]:

        distance_x_pixels, distance_y_pixels = measure_xy_distance(
            object_position, closest_key_point
        )

        distance_x_meters = convert_pixel_distance_to_meters(
            distance_x_pixels, player_height_meters, player_height_pixels
        )
        distance_y_meters = convert_pixel_distance_to_meters(
            distance_y_pixels, player_height_meters, player_height_pixels
        )

        mini_x = self.convert_meters_to_pixels(distance_x_meters)
        mini_y = self.convert_meters_to_pixels(distance_y_meters)

        mini_key_x = self.drawing_key_points[closest_key_point_index * 2]
        mini_key_y = self.drawing_key_points[closest_key_point_index * 2 + 1]

        return mini_key_x + mini_x, mini_key_y + mini_y

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes: List[Dict[int, Tuple[int, int, int, int]]],
        ball_boxes: List[Dict[int, Tuple[int, int, int, int]]],
        original_court_key_points: List[int],
    ) -> Tuple[List[Dict[int, Tuple[int, int]]], List[Dict[int, Tuple[int, int]]]]:

        logger.info("[MiniCourt] Converting bounding boxes...")

        output_player_boxes = []
        output_ball_boxes = []

        DEFAULT_PLAYER_HEIGHT = 1.75  # meters
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
        }

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num].get(1)
            ball_position = get_center_of_bbox(ball_box) if ball_box else None

            frame_output = {}

            for player_id in [1, 2]:
                bbox = player_bbox.get(player_id)
                if bbox:
                    foot = get_foot_position(bbox)
                    keypoint_idx = get_closest_keypoint_index(
                        foot, original_court_key_points, [0, 2, 12, 13]
                    )
                    keypoint = (
                        original_court_key_points[keypoint_idx * 2],
                        original_court_key_points[keypoint_idx * 2 + 1],
                    )

                    frame_window = range(
                        max(0, frame_num - 20), min(len(player_boxes), frame_num + 50)
                    )
                    heights = [
                        get_height_of_bbox(player_boxes[i].get(player_id, bbox))
                        for i in frame_window
                        if player_id in player_boxes[i]
                    ]
                    max_height = max(heights, default=get_height_of_bbox(bbox))

                    player_height = player_heights.get(player_id, DEFAULT_PLAYER_HEIGHT)
                    mini_pos = self.get_mini_court_coordinates(
                        foot, keypoint, keypoint_idx, max_height, player_height
                    )
                    frame_output[player_id] = mini_pos
                else:
                    frame_output[player_id] = (0, 0)

            output_player_boxes.append(frame_output)

            if ball_position:
                closest_player_id = min(
                    frame_output.keys(),
                    key=lambda pid: measure_distance(ball_position, frame_output[pid]),
                )
                output_ball_boxes.append(
                    {1: frame_output.get(closest_player_id, (0, 0))}
                )
            else:
                output_ball_boxes.append({1: (0, 0)})

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(
        self,
        frames: List[np.ndarray],
        positions: List[Dict[int, Tuple[int, int]]],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> List[np.ndarray]:

        logger.info("[MiniCourt] Drawing points on mini court...")

        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions):
                for pos in positions[frame_num].values():
                    x, y = map(int, pos)
                    cv2.circle(frame, (x, y), 5, color, -1)

        return frames
