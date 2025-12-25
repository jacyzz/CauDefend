import { Checkbox, Divider, Input } from 'antd';
import { useMemo, useState } from 'react';
import type { StyleItem } from '../api/ist';

type Props = {
  styles: StyleItem[];
  value: string[];
  onChange: (codes: string[]) => void;
};

export default function StylePicker({ styles, value, onChange }: Props) {
  const [query, setQuery] = useState('');
  const grouped = useMemo(() => {
    const map = new Map<string, StyleItem[]>();
    for (const s of styles) {
      const fam = s.family || 'others';
      if (!map.has(fam)) map.set(fam, []);
      map.get(fam)!.push(s);
    }
    for (const [, arr] of map) {
      arr.sort((a, b) => a.code.localeCompare(b.code));
    }
    return Array.from(map.entries()).sort((a, b) => {
      const ai = Number.isFinite(Number(a[0])) ? Number(a[0]) : 999;
      const bi = Number.isFinite(Number(b[0])) ? Number(b[0]) : 999;
      return ai - bi;
    });
  }, [styles]);

  const filteredCodes = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return styles.map((s) => s.code);
    return styles.filter((s) => s.code.includes(q) || s.type.includes(q) || s.subtype.includes(q)).map((s) => s.code);
  }, [styles, query]);

  const handleGroupChange = (groupCodes: string[]) => (codes: Array<string | number>) => {
    // Merge selection for this group with existing selections from other groups
    const selectedInGroup = (codes || []).filter((x): x is string => typeof x === 'string');
    const other = (value || []).filter((c) => !groupCodes.includes(c));
    const next = Array.from(new Set([...other, ...selectedInGroup]));
    onChange(next);
  };

  return (
    <div>
      <Input.Search
        placeholder="按 code/type/subtype 过滤"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ marginBottom: 8 }}
      />
      <div style={{ maxHeight: 320, overflowY: 'auto', border: '1px solid #eee', padding: 8 }}>
        {grouped.map(([fam, arr]) => {
          const options = arr
            .filter((s) => filteredCodes.includes(s.code))
            .map((s) => ({ label: `${s.code} (${s.subtype})`, value: s.code }));
          if (options.length === 0) return null;
          const codesInGroup = arr.map((s) => s.code);
          const groupValue = (value || []).filter((c) => codesInGroup.includes(c));
          return (
            <div key={fam} style={{ marginBottom: 8 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Group {fam}</div>
              <Checkbox.Group
                options={options}
                value={groupValue}
                onChange={handleGroupChange(codesInGroup)}
                style={{ width: '100%' }}
              />
              <Divider style={{ margin: '8px 0' }} />
            </div>
          );
        })}
      </div>
    </div>
  );
}


