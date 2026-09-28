#!/bin/bash
cd /home/data/projects/ai_factory

echo "================================"
echo "C3 Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================"

echo ""
echo "Current progress:"
docker exec -it postgres psql -U woami -d ai_factory -t -c "SELECT '  Vol ' || current_volume || ' Ch ' || current_chapter || ' Scene ' || current_scene || ' Completed=' || chapter_completed FROM writing_progress WHERE project_id = 'simple_long_novel_001';"

echo ""
echo "B2-2C2 stats:"
TRIGGERED=$(grep -c '\[B2-2C2\]' logs/ai_factory.log 2>/dev/null || echo 0)
RESCUED=$(grep -c 'Rescue applied' logs/ai_factory.log 2>/dev/null || echo 0)
echo "  Triggered: $TRIGGERED"
echo "  Rescued: $RESCUED"
if [ "$TRIGGERED" -gt 0 ]; then
    RATE=$(echo "scale=1; $RESCUED * 100 / $TRIGGERED" | bc)
    echo "  Rescue rate: ${RATE}%"
fi

echo ""
echo "Recent 5 rescues:"
grep "Rescue applied" logs/ai_factory.log 2>/dev/null | tail -5 | sed 's/.*"message": "//' | sed 's/"}//'

echo ""
echo "Chapters generated:"
ls data/novels/simple_long_novel_001/vol_001/chap_*.txt 2>/dev/null | wc -l

echo "================================"
