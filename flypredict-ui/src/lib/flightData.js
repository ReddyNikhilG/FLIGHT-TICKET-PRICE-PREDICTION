// Flight data options based on the training dataset
export const AIRLINES = [
  'Air India', 'AirAsia', 'GO FIRST', 'Indigo', 'SpiceJet', 'Vistara'
];

export const CITIES = [
  'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'
];

export const DEPARTURE_TIMES = [
  'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'
];

export const ARRIVAL_TIMES = [
  'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'
];

export const STOPS = ['zero', 'one', 'two_or_more'];

export const CLASSES = ['Economy', 'Business'];

export const FEATURE_IMPORTANCE = [
  { feature: 'Class (Economy)', importance: 0.5125 },
  { feature: 'Class (Business)', importance: 0.3782 },
  { feature: 'Duration', importance: 0.0515 },
  { feature: 'Days Left', importance: 0.0154 },
  { feature: 'Air India', importance: 0.0080 },
  { feature: 'Vistara', importance: 0.0031 },
  { feature: 'Source: Delhi', importance: 0.0027 },
  { feature: 'Dest: Delhi', importance: 0.0026 },
  { feature: 'Dest: Mumbai', importance: 0.0024 },
  { feature: 'Dest: Kolkata', importance: 0.0020 },
  { feature: 'Source: Mumbai', importance: 0.0019 },
  { feature: 'Source: Kolkata', importance: 0.0018 },
  { feature: 'Dest: Hyderabad', importance: 0.0017 },
  { feature: 'Source: Hyderabad', importance: 0.0016 },
  { feature: 'Arrival: Evening', importance: 0.0015 },
];

export const MODEL_METRICS = {
  r2: 0.9678,
  mae: 2313.27,
  rmse: 4116.44,
  cv_r2_mean: 0.9685,
  cv_r2_std: 0.0013,
  train_size: 18008,
  test_size: 4503,
};

export const MODEL_COMPARISON = [
  { model: 'Linear Regression', accuracy: 73, r2: 0.73, color: 'hsl(var(--chart-5))' },
  { model: 'Decision Tree', accuracy: 90, r2: 0.90, color: 'hsl(var(--chart-2))' },
  { model: 'Random Forest', accuracy: 93, r2: 0.97, color: 'hsl(var(--chart-1))' },
];

export function formatTimeLabel(time) {
  return time.replace(/_/g, ' ');
}

export function formatStopsLabel(stops) {
  if (stops === 'zero' || stops === 0) return 'Non-stop';
  if (stops === 'one' || stops === 1) return '1 Stop';
  return '2+ Stops';
}
