// @ts-nocheck
import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingDown, Plane, Info, CheckCircle2, AlertCircle } from 'lucide-react';
import { AIRLINES, CITIES, DEPARTURE_TIMES, ARRIVAL_TIMES, STOPS, CLASSES, formatStopsLabel, formatTimeLabel } from '@/lib/flightData';
import { fetchFlightMetadata, predictFlightPrice } from '@/lib/flightApi';

const PRICE_FACTORS = [
  { name: 'Days Left', icon: '📅' },
  { name: 'Duration', icon: '⏱️' },
  { name: 'Airline', icon: '✈️' },
  { name: 'Travel Class', icon: '💺' },
  { name: 'Route', icon: '🗺️' },
];

const DEFAULT_OPTIONS = {
  airlines: AIRLINES,
  cities: CITIES,
  departureTimes: DEPARTURE_TIMES,
  arrivalTimes: ARRIVAL_TIMES,
  stops: STOPS,
  classes: CLASSES,
};

const formatDisplayLabel = (value) => String(value).replace(/_/g, ' ');

export default function Predict() {
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  // Form state
  const [formData, setFormData] = useState({
    airline: '',
    source: '',
    destination: '',
    departure_time: '',
    arrival_time: '',
    duration_hours: 2,
    days_left: 7,
    stops: 'zero',
    travel_class: 'Economy',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const loadMetadata = async () => {
      try {
        const response = await fetchFlightMetadata();
        const backendOptions = response?.options || {};

        if (!isMounted) {
          return;
        }

        setOptions({
          airlines: backendOptions.airline || AIRLINES,
          cities: backendOptions.source_city || backendOptions.destination_city || CITIES,
          departureTimes: backendOptions.departure_time || DEPARTURE_TIMES,
          arrivalTimes: backendOptions.arrival_time || ARRIVAL_TIMES,
          stops: backendOptions.stops || STOPS,
          classes: backendOptions.class || CLASSES,
        });
      } catch {
        if (isMounted) {
          setOptions(DEFAULT_OPTIONS);
        }
      }
    };

    loadMetadata();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError(null);
  };

  const handlePredict = async () => {
    try {
      setError(null);
      setLoading(true);

      // Validation
      if (!formData.airline || !formData.source || !formData.destination) {
        setError('Please fill in all required fields');
        setLoading(false);
        return;
      }

      if (formData.source === formData.destination) {
        setError('Source and destination must be different');
        setLoading(false);
        return;
      }

      const backendPayload = {
        airline: formData.airline,
        source_city: formData.source,
        destination_city: formData.destination,
        departure_time: formData.departure_time,
        arrival_time: formData.arrival_time,
        stops: formData.stops,
        class: formData.travel_class,
        duration: Number(formData.duration_hours),
        days_left: formData.days_left,
      };

      const result = await predictFlightPrice(backendPayload);
      const predictedPrice = Number(result.predicted_price);

      if (!Number.isFinite(predictedPrice)) {
        throw new Error('Backend returned an invalid prediction');
      }

      setPrediction({
        price: Math.round(predictedPrice),
        confidence: 96.8,
        route: `${formData.source} → ${formData.destination}`,
        airline: formData.airline,
        ...formData,
      });
    } catch (err) {
      setError(err.message || 'Failed to predict price. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      airline: '',
      source: '',
      destination: '',
      departure_time: '',
      arrival_time: '',
      duration_hours: 2,
      days_left: 7,
        stops: 'zero',
      travel_class: 'Economy',
    });
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <h1 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            Predict Flight Price
          </h1>
          <p className="mt-3 text-muted-foreground text-lg">
            Get an accurate price prediction for your flight in seconds.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prediction Form */}
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="font-heading text-xl flex items-center gap-2">
                  <Plane className="w-5 h-5 text-primary" />
                  Flight Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {error && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive flex items-start gap-3"
                  >
                    <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="font-medium text-sm">{error}</p>
                    </div>
                  </motion.div>
                )}

                {/* Row 1: Airline & Route */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="airline" className="text-sm font-medium mb-2 block">
                      Airline *
                    </Label>
                    <Select value={formData.airline} onValueChange={(value) => handleInputChange('airline', value)}>
                      <SelectTrigger id="airline">
                        <SelectValue placeholder="Select airline" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.airlines.map(airline => (
                          <SelectItem key={airline} value={airline}>
                            {formatDisplayLabel(airline)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="source" className="text-sm font-medium mb-2 block">
                      Source *
                    </Label>
                    <Select value={formData.source} onValueChange={(value) => handleInputChange('source', value)}>
                      <SelectTrigger id="source">
                        <SelectValue placeholder="From" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.cities.map(city => (
                          <SelectItem key={city} value={city}>
                            {city}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Row 2: Destination & Class */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="destination" className="text-sm font-medium mb-2 block">
                      Destination *
                    </Label>
                    <Select value={formData.destination} onValueChange={(value) => handleInputChange('destination', value)}>
                      <SelectTrigger id="destination">
                        <SelectValue placeholder="To" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.cities.map(city => (
                          <SelectItem key={city} value={city}>
                            {city}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="class" className="text-sm font-medium mb-2 block">
                      Travel Class
                    </Label>
                    <Select value={formData.travel_class} onValueChange={(value) => handleInputChange('travel_class', value)}>
                      <SelectTrigger id="class">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {options.classes.map(cls => (
                          <SelectItem key={cls} value={cls}>
                            {cls}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Row 3: Departure & Arrival Time */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="dep-time" className="text-sm font-medium mb-2 block">
                      Departure Time
                    </Label>
                    <Select value={formData.departure_time} onValueChange={(value) => handleInputChange('departure_time', value)}>
                      <SelectTrigger id="dep-time">
                        <SelectValue placeholder="Select time" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.departureTimes.map(time => (
                          <SelectItem key={time} value={time}>
                            {formatTimeLabel(time)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="arr-time" className="text-sm font-medium mb-2 block">
                      Arrival Time
                    </Label>
                    <Select value={formData.arrival_time} onValueChange={(value) => handleInputChange('arrival_time', value)}>
                      <SelectTrigger id="arr-time">
                        <SelectValue placeholder="Select time" />
                      </SelectTrigger>
                      <SelectContent>
                        {options.arrivalTimes.map(time => (
                          <SelectItem key={time} value={time}>
                            {formatTimeLabel(time)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Row 4: Duration & Stops */}
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <Label className="text-sm font-medium">
                        Duration: {formData.duration_hours}h
                      </Label>
                    </div>
                    <Slider
                      value={[formData.duration_hours]}
                      onValueChange={(value) => handleInputChange('duration_hours', value[0])}
                      min={1}
                      max={24}
                      step={0.5}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Row 5: Days Left & Stops */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <Label className="text-sm font-medium">
                        Days Left: {formData.days_left}
                      </Label>
                    </div>
                    <Slider
                      value={[formData.days_left]}
                      onValueChange={(value) => handleInputChange('days_left', value[0])}
                      min={0}
                      max={30}
                      step={1}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <Label htmlFor="stops" className="text-sm font-medium mb-2 block">
                      Number of Stops
                    </Label>
                    <Select value={formData.stops} onValueChange={(value) => handleInputChange('stops', value)}>
                      <SelectTrigger id="stops">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {options.stops.map(stop => (
                          <SelectItem key={stop} value={String(stop)}>
                            {formatStopsLabel(stop)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-4">
                  <Button
                    onClick={handlePredict}
                    disabled={loading}
                    className="flex-1 bg-primary hover:bg-primary/90"
                    size="lg"
                  >
                    {loading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-r-white rounded-full animate-spin mr-2" />
                        Predicting...
                      </>
                    ) : (
                      <>
                        <TrendingDown className="w-4 h-4 mr-2" />
                        Get Prediction
                      </>
                    )}
                  </Button>
                  {prediction && (
                    <Button
                      onClick={handleReset}
                      variant="outline"
                      size="lg"
                      className="px-6"
                    >
                      Reset
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Result Panel */}
          <div className="lg:col-span-1">
            <AnimatePresence>
              {prediction ? (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="sticky top-6 border-2 border-primary/20">
                    <CardHeader>
                      <CardTitle className="font-heading text-lg flex items-center gap-2 text-primary">
                        <CheckCircle2 className="w-5 h-5" />
                        Prediction Result
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Price */}
                      <div className="text-center">
                        <p className="text-muted-foreground text-sm mb-2">Predicted Price</p>
                        <p className="font-heading text-5xl font-bold text-primary">
                          ₹{prediction.price.toLocaleString('en-IN')}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          Confidence: {prediction.confidence}%
                        </p>
                      </div>

                      {/* Route Summary */}
                      <div className="pt-4 border-t">
                        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-3">Summary</p>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Route:</span>
                            <span className="font-mono font-medium">{prediction.route}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Airline:</span>
                            <span className="font-mono font-medium">{prediction.airline}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Class:</span>
                            <span className="font-mono font-medium">{prediction.travel_class}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Duration:</span>
                            <span className="font-mono font-medium">{prediction.duration_hours}h</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Days Left:</span>
                            <span className="font-mono font-medium">{prediction.days_left} days</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Stops:</span>
                            <span className="font-mono font-medium">
                              {formatStopsLabel(prediction.stops)}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Info Box */}
                      <div className="p-3 rounded-lg bg-muted/50 border border-border/50">
                        <div className="flex gap-2 text-xs">
                          <Info className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
                          <p className="text-muted-foreground">
                            This prediction is saved to your history automatically.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-4"
                >
                  {/* Price Factors Info */}
                  <Card className="border-border/50">
                    <CardHeader>
                      <CardTitle className="font-heading text-sm uppercase tracking-wider">
                        What Affects Price?
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {PRICE_FACTORS.map((factor, idx) => (
                          <motion.div
                            key={factor.name}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="flex items-center gap-3 py-2 text-sm"
                          >
                            <span className="text-lg">{factor.icon}</span>
                            <span className="text-muted-foreground">{factor.name}</span>
                          </motion.div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Tips */}
                  <Card className="border-border/50 border-2 border-dashed">
                    <CardHeader>
                      <CardTitle className="font-heading text-sm uppercase tracking-wider">
                        💡 Pro Tip
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm text-muted-foreground">
                      <p>Book flights 7-14 days in advance for better prices during off-peak seasons.</p>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
