/** Court model in metres: x 0..10 across, y 0..20 along, origin at the near-left corner, net at y = 10. */

export type Pt = [number, number];

export const POINT_NAMES = [
  'bottom_left', 'bottom_right', 'top_left', 'top_right', 'net_left', 'net_right',
  'service_near_left', 'service_near_center', 'service_near_right',
  'service_far_left', 'service_far_center', 'service_far_right',
] as const;

export type PointName = (typeof POINT_NAMES)[number];

/** Court position of each point, in the API.md clicking order. */
export const COURT_POINTS: Record<PointName, Pt> = {
  bottom_left: [0, 0],
  bottom_right: [10, 0],
  top_left: [0, 20],
  top_right: [10, 20],
  net_left: [0, 10],
  net_right: [10, 10],
  service_near_left: [0, 3],
  service_near_center: [5, 3],
  service_near_right: [10, 3],
  service_far_left: [0, 17],
  service_far_center: [5, 17],
  service_far_right: [10, 17],
};

export const POINT_LABELS: Record<PointName, string> = {
  bottom_left: 'Near baseline, left corner',
  bottom_right: 'Near baseline, right corner',
  top_left: 'Far baseline, left corner',
  top_right: 'Far baseline, right corner',
  net_left: 'Net, left end',
  net_right: 'Net, right end',
  service_near_left: 'Near service line, left end',
  service_near_center: 'Near service line, centre',
  service_near_right: 'Near service line, right end',
  service_far_left: 'Far service line, left end',
  service_far_center: 'Far service line, centre',
  service_far_right: 'Far service line, right end',
};

/** Outline, net, the two service lines and the centre line between them. */
export const COURT_LINES: [Pt, Pt][] = [
  [[0, 0], [10, 0]],
  [[10, 0], [10, 20]],
  [[10, 20], [0, 20]],
  [[0, 20], [0, 0]],
  [[0, 10], [10, 10]],
  [[0, 3], [10, 3]],
  [[0, 17], [10, 17]],
  [[5, 3], [5, 17]],
];

export function isPointName(name: string): name is PointName {
  return (POINT_NAMES as readonly string[]).includes(name);
}
