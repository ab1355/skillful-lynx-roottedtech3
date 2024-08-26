class NotificationSystem {
    constructor() {
        this.container = document.createElement('div');
        this.container.id = 'notification-container';
        document.body.appendChild(this.container);
    }

    show(message, type = 'info') {
        if (!userSettings.getSettings().notificationsEnabled) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        const closeBtn = document.createElement('button');
        closeBtn.textContent = '×';
        closeBtn.className = 'close-btn';
        closeBtn.onclick = () => this.container.removeChild(notification);

        notification.appendChild(closeBtn);
        this.container.appendChild(notification);

        setTimeout(() => {
            if (this.container.contains(notification)) {
                this.container.removeChild(notification);
            }
        }, 5000);
    }
}

const notificationSystem = new NotificationSystem();

// Example usage:
// notificationSystem.show('Welcome to AgentZero HR Analytics!', 'info');
// notificationSystem.show('Error occurred while processing data.', 'error');
// notificationSystem.show('Successfully updated user profile.', 'success');