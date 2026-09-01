//! ORCA / RVO2 local collision avoidance.
//!
//! A direct port of `sim/orca.py`, which itself follows the reference RVO2
//! implementation: one half-plane per neighbour, then a 2D linear program for
//! the velocity closest to the preferred one.

pub const EPS: f32 = 1e-5;

#[derive(Clone, Copy)]
pub struct Line {
    pub px: f32,
    pub py: f32,
    pub dx: f32,
    pub dy: f32,
}

#[inline]
fn det(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    ax * by - ay * bx
}

#[inline]
fn norm(x: f32, y: f32) -> (f32, f32) {
    let n = (x * x + y * y).sqrt();
    if n < EPS {
        (0.0, 0.0)
    } else {
        (x / n, y / n)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn orca_line(
    pos: (f32, f32),
    vel: (f32, f32),
    radius: f32,
    other_pos: (f32, f32),
    other_vel: (f32, f32),
    other_radius: f32,
    time_horizon: f32,
    dt: f32,
    share: f32,
) -> Line {
    let (rpx, rpy) = (other_pos.0 - pos.0, other_pos.1 - pos.1);
    let (rvx, rvy) = (vel.0 - other_vel.0, vel.1 - other_vel.1);
    let dist_sq = rpx * rpx + rpy * rpy;
    let r = radius + other_radius;
    let r_sq = r * r;

    let (dirx, diry, ux, uy);

    if dist_sq > r_sq {
        let inv_h = 1.0 / time_horizon;
        let (wx, wy) = (rvx - inv_h * rpx, rvy - inv_h * rpy);
        let w_len_sq = wx * wx + wy * wy;
        let dot1 = wx * rpx + wy * rpy;
        if dot1 < 0.0 && dot1 * dot1 > r_sq * w_len_sq {
            // closest point is on the cut-off circle
            let w_len = w_len_sq.sqrt();
            let (nx, ny) = (wx / w_len, wy / w_len);
            dirx = ny;
            diry = -nx;
            let f = r * inv_h - w_len;
            ux = f * nx;
            uy = f * ny;
        } else {
            // closest point is on one of the legs
            let leg = (dist_sq - r_sq).max(0.0).sqrt();
            if det(rpx, rpy, wx, wy) > 0.0 {
                dirx = (rpx * leg - rpy * r) / dist_sq;
                diry = (rpx * r + rpy * leg) / dist_sq;
            } else {
                dirx = -(rpx * leg + rpy * r) / dist_sq;
                diry = -(-rpx * r + rpy * leg) / dist_sq;
            }
            let dot2 = rvx * dirx + rvy * diry;
            ux = dot2 * dirx - rvx;
            uy = dot2 * diry - rvy;
        }
    } else {
        // already overlapping: push apart over one timestep
        let inv_dt = 1.0 / dt;
        let (wx, wy) = (rvx - inv_dt * rpx, rvy - inv_dt * rpy);
        let w_len = (wx * wx + wy * wy).sqrt();
        if w_len < EPS {
            return Line { px: vel.0, py: vel.1, dx: 1.0, dy: 0.0 };
        }
        let (nx, ny) = (wx / w_len, wy / w_len);
        dirx = ny;
        diry = -nx;
        let f = r * inv_dt - w_len;
        ux = f * nx;
        uy = f * ny;
    }

    Line { px: vel.0 + share * ux, py: vel.1 + share * uy, dx: dirx, dy: diry }
}

fn lp1(
    lines: &[Line],
    i: usize,
    radius: f32,
    opt: (f32, f32),
    direction_opt: bool,
) -> Option<(f32, f32)> {
    let ln = lines[i];
    let dot = ln.px * ln.dx + ln.py * ln.dy;
    let disc = dot * dot + radius * radius - (ln.px * ln.px + ln.py * ln.py);
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let mut t_left = -dot - sq;
    let mut t_right = -dot + sq;

    for lj in lines.iter().take(i) {
        let denom = det(ln.dx, ln.dy, lj.dx, lj.dy);
        let numer = det(lj.dx, lj.dy, ln.px - lj.px, ln.py - lj.py);
        if denom.abs() <= EPS {
            if numer < 0.0 {
                return None;
            }
            continue;
        }
        let t = numer / denom;
        if denom >= 0.0 {
            t_right = t_right.min(t);
        } else {
            t_left = t_left.max(t);
        }
        if t_left > t_right {
            return None;
        }
    }

    let t = if direction_opt {
        if opt.0 * ln.dx + opt.1 * ln.dy > 0.0 { t_right } else { t_left }
    } else {
        (ln.dx * (opt.0 - ln.px) + ln.dy * (opt.1 - ln.py)).clamp(t_left, t_right)
    };
    Some((ln.px + t * ln.dx, ln.py + t * ln.dy))
}

fn lp2(
    lines: &[Line],
    radius: f32,
    opt: (f32, f32),
    direction_opt: bool,
) -> ((f32, f32), usize) {
    let mut result = if direction_opt {
        (opt.0 * radius, opt.1 * radius)
    } else if opt.0 * opt.0 + opt.1 * opt.1 > radius * radius {
        let (nx, ny) = norm(opt.0, opt.1);
        (nx * radius, ny * radius)
    } else {
        opt
    };

    for (i, ln) in lines.iter().enumerate() {
        if det(ln.dx, ln.dy, ln.px - result.0, ln.py - result.1) > 0.0 {
            match lp1(lines, i, radius, opt, direction_opt) {
                Some(v) => result = v,
                None => return (result, i),
            }
        }
    }
    (result, lines.len())
}

fn lp3(lines: &[Line], begin: usize, radius: f32, mut result: (f32, f32)) -> (f32, f32) {
    let mut distance = 0.0f32;
    for i in begin..lines.len() {
        let li = lines[i];
        if det(li.dx, li.dy, li.px - result.0, li.py - result.1) <= distance {
            continue;
        }
        let mut proj: Vec<Line> = Vec::with_capacity(i);
        for lj in lines.iter().take(i) {
            let d = det(li.dx, li.dy, lj.dx, lj.dy);
            let (px, py);
            if d.abs() <= EPS {
                if li.dx * lj.dx + li.dy * lj.dy > 0.0 {
                    continue;
                }
                px = 0.5 * (li.px + lj.px);
                py = 0.5 * (li.py + lj.py);
            } else {
                let t = det(lj.dx, lj.dy, li.px - lj.px, li.py - lj.py) / d;
                px = li.px + t * li.dx;
                py = li.py + t * li.dy;
            }
            let (dx, dy) = norm(lj.dx - li.dx, lj.dy - li.dy);
            proj.push(Line { px, py, dx, dy });
        }
        let (new, idx) = lp2(&proj, radius, (-li.dy, li.dx), true);
        if idx >= proj.len() {
            result = new;
        }
        distance = det(li.dx, li.dy, li.px - result.0, li.py - result.1);
    }
    result
}

/// Collision-free velocity closest to `pref_vel`.
/// `neighbours` is (pos, vel, radius); the nearest `max_neighbours` are used.
pub fn new_velocity(
    pos: (f32, f32),
    vel: (f32, f32),
    radius: f32,
    max_speed: f32,
    pref_vel: (f32, f32),
    neighbours: &mut Vec<((f32, f32), (f32, f32), f32)>,
    time_horizon: f32,
    dt: f32,
    max_neighbours: usize,
    share: f32,
) -> (f32, f32) {
    if neighbours.len() > max_neighbours {
        neighbours.sort_by(|a, b| {
            let da = (a.0 .0 - pos.0).powi(2) + (a.0 .1 - pos.1).powi(2);
            let db = (b.0 .0 - pos.0).powi(2) + (b.0 .1 - pos.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbours.truncate(max_neighbours);
    }

    let lines: Vec<Line> = neighbours
        .iter()
        .map(|&(np, nv, nr)| {
            orca_line(pos, vel, radius, np, nv, nr, time_horizon, dt, share)
        })
        .collect();

    let (result, idx) = lp2(&lines, max_speed, pref_vel, false);
    if idx < lines.len() {
        lp3(&lines, idx, max_speed, result)
    } else {
        result
    }
}
