// @ts-nocheck
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { motion } from 'framer-motion';
import { formatDistanceToNow } from 'date-fns';
import { Calendar, AlertCircle, Clock } from 'lucide-react';
import { fetchPredictionHistory } from '@/lib/flightApi';
import { formatStopsLabel } from '@/lib/flightData';

const getRouteValue = (item) => `${item.source_city || item.source || ''}${item.destination_city || item.destination || ''}`;
const getRouteLabel = (item) => `${item.source_city || item.source || '—'} → ${item.destination_city || item.destination || '—'}`;
const getClassValue = (item) => item.class || item.travel_class || '—';
const getDurationValue = (item) => item.duration ?? item.duration_hours;
const formatDisplayLabel = (value) => String(value).replace(/_/g, ' ');

export default function History() {
  const [sortBy, setSortBy] = useState('date');
  const [filterAirline, setFilterAirline] = useState('all');

  // Fetch prediction history
  const { data: predictions = [], isLoading, error } = useQuery({
    queryKey: ['predictions'],
    queryFn: async () => {
      const result = await fetchPredictionHistory();
      const records = result.history || [];
      return records.sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Filter predictions
  const filteredPredictions = filterAirline === 'all' 
    ? predictions 
    : predictions.filter(p => p.airline?.toLowerCase() === filterAirline.toLowerCase());

  // Sort predictions
  const sortedPredictions = [...filteredPredictions].sort((a, b) => {
    switch (sortBy) {
      case 'price':
        return b.predicted_price - a.predicted_price;
      case 'route':
        return getRouteValue(a).localeCompare(getRouteValue(b));
      default:
        return new Date(b.timestamp) - new Date(a.timestamp);
    }
  });

  // Get unique airlines
  const airlines = [...new Set(predictions.map(p => p.airline))].sort();

  const priceColor = (actual, predicted) => {
    const diff = ((predicted - actual) / actual) * 100;
    if (diff > 10) return 'text-destructive';
    if (diff > 5) return 'text-orange-500';
    if (diff > -5) return 'text-primary';
    return 'text-green-500';
  };

  const getAccuracyBadge = (actual, predicted) => {
    if (!actual) return <Badge>N/A</Badge>;
    const error = Math.abs(((predicted - actual) / actual) * 100);
    if (error < 5) return <Badge className="bg-green-500/20 text-green-700">Excellent</Badge>;
    if (error < 10) return <Badge className="bg-blue-500/20 text-blue-700">Good</Badge>;
    if (error < 20) return <Badge className="bg-orange-500/20 text-orange-700">Fair</Badge>;
    return <Badge className="bg-destructive/20 text-destructive">Poor</Badge>;
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            Prediction History
          </h1>
          <p className="mt-3 text-muted-foreground text-lg">
            View and analyze all past price predictions.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1">
            <label className="text-sm text-muted-foreground mb-2 block">Sort by</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="date">Most Recent</option>
              <option value="price">Highest Price</option>
              <option value="route">Route</option>
            </select>
          </div>
          <div className="flex-1">
            <label className="text-sm text-muted-foreground mb-2 block">Filter by Airline</label>
            <select
              value={filterAirline}
              onChange={(e) => setFilterAirline(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="all">All Airlines</option>
              {airlines.map(airline => (
                <option key={airline} value={airline}>{airline}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Table */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="font-heading text-lg flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-primary" />
                  {sortedPredictions.length} Prediction{sortedPredictions.length !== 1 ? 's' : ''}
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive flex items-center gap-2 mb-6">
                  <AlertCircle className="w-5 h-5" />
                  Failed to load prediction history
                </div>
              )}

              {isLoading ? (
                <div className="space-y-3">
                  {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : sortedPredictions.length === 0 ? (
                <div className="py-12 text-center">
                  <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                  <p className="text-muted-foreground">No predictions yet</p>
                  <p className="text-sm text-muted-foreground mt-1">Start by making your first prediction</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Route</TableHead>
                        <TableHead>Airline</TableHead>
                        <TableHead>Class</TableHead>
                        <TableHead>Stops</TableHead>
                        <TableHead>Duration</TableHead>
                        <TableHead>Predicted Price</TableHead>
                        <TableHead>Actual Price</TableHead>
                        <TableHead>Accuracy</TableHead>
                        <TableHead>Time</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedPredictions.map((pred, idx) => (
                        <motion.tr
                          key={idx}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: idx * 0.05 }}
                          className="border-b hover:bg-muted/50 transition-colors"
                        >
                          <TableCell className="font-medium">
                            {getRouteLabel(pred)}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">{formatDisplayLabel(pred.airline)}</Badge>
                          </TableCell>
                          <TableCell className="text-sm">{getClassValue(pred)}</TableCell>
                          <TableCell className="text-sm">{formatStopsLabel(pred.stops ?? 'zero')}</TableCell>
                          <TableCell className="text-sm">
                            {getDurationValue(pred) ? `${getDurationValue(pred)}h` : '—'}
                          </TableCell>
                          <TableCell>
                            <span className="font-mono font-bold text-primary">
                              ₹{Math.round(pred.predicted_price).toLocaleString('en-IN')}
                            </span>
                          </TableCell>
                          <TableCell>
                            {pred.actual_price ? (
                              <span className={`font-mono font-bold ${priceColor(pred.actual_price, pred.predicted_price)}`}>
                                ₹{Math.round(pred.actual_price).toLocaleString('en-IN')}
                              </span>
                            ) : (
                              <span className="text-muted-foreground text-sm">—</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {getAccuracyBadge(pred.actual_price, pred.predicted_price)}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {pred.timestamp ? formatDistanceToNow(new Date(pred.timestamp), { addSuffix: true }) : '—'}
                          </TableCell>
                        </motion.tr>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Summary Stats */}
        {sortedPredictions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8"
          >
            <Card>
              <CardContent className="pt-6">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Average Prediction</p>
                <p className="font-heading text-2xl font-bold text-primary">
                  ₹{Math.round(
                    sortedPredictions.reduce((sum, p) => sum + p.predicted_price, 0) / sortedPredictions.length
                  ).toLocaleString('en-IN')}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Highest Prediction</p>
                <p className="font-heading text-2xl font-bold text-chart-3">
                  ₹{Math.round(Math.max(...sortedPredictions.map(p => p.predicted_price))).toLocaleString('en-IN')}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Lowest Prediction</p>
                <p className="font-heading text-2xl font-bold text-accent">
                  ₹{Math.round(Math.min(...sortedPredictions.map(p => p.predicted_price))).toLocaleString('en-IN')}
                </p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}
