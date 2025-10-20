#!/bin/bash
# MLflow Service Management Script
# Usage: ./manage_mlflow_service.sh [start|stop|restart|status|logs]

set -e

SERVICE_NAME="mlflow"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
    start)
        echo "🚀 Starting MLflow service..."
        sudo systemctl start "$SERVICE_NAME"
        echo "✅ MLflow service started"
        ;;
    stop)
        echo "🛑 Stopping MLflow service..."
        sudo systemctl stop "$SERVICE_NAME"
        echo "✅ MLflow service stopped"
        ;;
    restart)
        echo "🔄 Restarting MLflow service..."
        sudo systemctl restart "$SERVICE_NAME"
        echo "✅ MLflow service restarted"
        ;;
    status)
        echo "📊 MLflow service status:"
        sudo systemctl status "$SERVICE_NAME" --no-pager
        ;;
    logs)
        echo "📋 MLflow service logs (last 50 lines):"
        sudo journalctl -u "$SERVICE_NAME" -n 50 --no-pager
        ;;
    enable)
        echo "🔧 Enabling MLflow service to start on boot..."
        sudo systemctl enable "$SERVICE_NAME"
        echo "✅ MLflow service enabled for auto-start"
        ;;
    disable)
        echo "🔧 Disabling MLflow service from auto-start..."
        sudo systemctl disable "$SERVICE_NAME"
        echo "✅ MLflow service disabled from auto-start"
        ;;
    *)
        echo "MLflow Service Management Script"
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "Commands:"
        echo "  start   - Start MLflow service"
        echo "  stop    - Stop MLflow service"
        echo "  restart - Restart MLflow service"
        echo "  status  - Show service status"
        echo "  logs    - Show service logs"
        echo "  enable  - Enable auto-start on boot"
        echo "  disable - Disable auto-start on boot"
        echo ""
        echo "Web UI: http://localhost:5000 or http://$(hostname -I | awk '{print $1}'):5000"
        exit 1
        ;;
esac