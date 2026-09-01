//! Grid, A* and waypoint following.
//!
//! Mirrors `sim/nav.py` exactly, including the octile heuristic, the
//! no-corner-cutting rule, and the `pick_next_waypoint_dist` follower, because
//! those choices measurably change trajectories even on an open grid.

use std::collections::BinaryHeap;

pub const DIAG: f32 = std::f32::consts::SQRT_2;
pub const NEIGHBOURS: [(i32, i32); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
];

pub struct Grid {
    pub rows: i32,
    pub cols: i32,
    pub tile: f32,
    pub walkable: Vec<bool>,
}

impl Grid {
    pub fn new(rows: i32, cols: i32, tile: f32, walkable: Vec<bool>) -> Self {
        Grid { rows, cols, tile, walkable }
    }

    #[inline]
    pub fn in_bounds(&self, r: i32, c: i32) -> bool {
        r >= 0 && r < self.rows && c >= 0 && c < self.cols
    }

    #[inline]
    pub fn is_walkable(&self, r: i32, c: i32) -> bool {
        self.in_bounds(r, c) && self.walkable[(r * self.cols + c) as usize]
    }

    #[inline]
    pub fn to_world(&self, r: i32, c: i32) -> (f32, f32) {
        ((c as f32 + 0.5) * self.tile, (r as f32 + 0.5) * self.tile)
    }

    #[inline]
    pub fn to_cell(&self, x: f32, y: f32) -> (i32, i32) {
        ((y / self.tile).floor() as i32, (x / self.tile).floor() as i32)
    }

    #[inline]
    pub fn clamp_world(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x.clamp(0.0, self.cols as f32 * self.tile - 1e-3),
            y.clamp(0.0, self.rows as f32 * self.tile - 1e-3),
        )
    }
}

#[derive(PartialEq)]
struct Entry(f32, f32, i32);

impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reversed: BinaryHeap is a max-heap, we want the smallest f first
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn octile(a: (i32, i32), b: (i32, i32)) -> f32 {
    let dr = (a.0 - b.0).abs() as f32;
    let dc = (a.1 - b.1).abs() as f32;
    (dr + dc) + (DIAG - 2.0) * dr.min(dc)
}

/// 8-connected A*. Returns cells from start to goal, or empty.
pub fn astar(grid: &Grid, start: (i32, i32), goal: (i32, i32)) -> Vec<(i32, i32)> {
    if start == goal {
        return vec![start];
    }
    if !grid.is_walkable(goal.0, goal.1) {
        return Vec::new();
    }
    // Python's A* keys a dict, so an out-of-bounds start just finds no
    // neighbours and yields an empty path; the caller then steers straight.
    // A flat array would wrap the index instead, so guard it explicitly.
    if !grid.is_walkable(start.0, start.1) {
        return Vec::new();
    }
    let n = (grid.rows * grid.cols) as usize;
    let idx = |r: i32, c: i32| (r * grid.cols + c) as usize;

    let mut cost = vec![f32::INFINITY; n];
    let mut came = vec![-1i32; n];
    let mut heap = BinaryHeap::new();

    cost[idx(start.0, start.1)] = 0.0;
    heap.push(Entry(octile(start, goal), 0.0, idx(start.0, start.1) as i32));

    while let Some(Entry(_, g, cur)) = heap.pop() {
        let (cr, cc) = (cur / grid.cols, cur % grid.cols);
        if (cr, cc) == goal {
            let mut path = Vec::new();
            let mut at = cur;
            while at >= 0 {
                path.push((at / grid.cols, at % grid.cols));
                at = came[at as usize];
            }
            path.reverse();
            return path;
        }
        if g > cost[cur as usize] {
            continue;
        }
        for (dr, dc) in NEIGHBOURS {
            let (nr, nc) = (cr + dr, cc + dc);
            if !grid.is_walkable(nr, nc) {
                continue;
            }
            if dr != 0 && dc != 0 {
                // no corner cutting, as A* Pathfinding Project does by default
                if !(grid.is_walkable(cr + dr, cc) && grid.is_walkable(cr, cc + dc)) {
                    continue;
                }
            }
            let step = if dr != 0 && dc != 0 { DIAG } else { 1.0 };
            let ng = g + step;
            let ni = idx(nr, nc);
            if ng < cost[ni] {
                cost[ni] = ng;
                came[ni] = cur;
                heap.push(Entry(ng + octile((nr, nc), goal), ng, ni as i32));
            }
        }
    }
    Vec::new()
}

pub struct Follower {
    pub waypoints: Vec<(f32, f32)>,
    pub index: usize,
    pub pick_next_dist: f32,
}

impl Follower {
    pub fn new(pick_next_dist: f32) -> Self {
        Follower { waypoints: Vec::new(), index: 0, pick_next_dist }
    }

    pub fn set_path(&mut self, grid: &Grid, cells: &[(i32, i32)]) {
        self.waypoints = cells.iter().map(|&(r, c)| grid.to_world(r, c)).collect();
        self.index = 0;
    }

    pub fn done(&self) -> bool {
        self.index >= self.waypoints.len()
    }

    /// Unit vector toward the current waypoint, advancing past reached ones.
    pub fn desired_direction(&mut self, x: f32, y: f32) -> (f32, f32) {
        while self.index < self.waypoints.len() {
            let (wx, wy) = self.waypoints[self.index];
            let (dx, dy) = (wx - x, wy - y);
            let d = (dx * dx + dy * dy).sqrt();
            if d <= self.pick_next_dist && self.index < self.waypoints.len() - 1 {
                self.index += 1;
                continue;
            }
            if d < 1e-6 {
                return (0.0, 0.0);
            }
            return (dx / d, dy / d);
        }
        (0.0, 0.0)
    }
}
