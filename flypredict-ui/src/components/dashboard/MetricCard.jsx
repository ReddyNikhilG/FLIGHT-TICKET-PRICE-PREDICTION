import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';

export default function MetricCard({ title, value, subtitle, icon: Icon, color, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <Card className="border-border/50 hover:shadow-md transition-shadow">
        <CardContent className="p-5">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</p>
              <p className={`font-heading text-3xl font-bold mt-2 ${color || 'text-primary'}`}>{value}</p>
              {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
            </div>
            {Icon && (
              <div className={`w-10 h-10 rounded-xl ${color ? color.replace('text-', 'bg-') + '/10' : 'bg-primary/10'} flex items-center justify-center`}>
                <Icon className={`w-5 h-5 ${color || 'text-primary'}`} />
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
