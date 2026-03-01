/**
 * Dashboard Page
 * 
 * Admin/Teacher dashboard showing:
 * - Summary statistics
 * - Today's attendance table
 * - All attendance records with date filter
 * - Registered students list
 */

import React, { useState, useEffect, useCallback } from 'react';
import { attendanceAPI, studentAPI } from '../services/api';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('today');
  const [todayAttendance, setTodayAttendance] = useState([]);
  const [allAttendance, setAllAttendance] = useState([]);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dateFilter, setDateFilter] = useState('');
  const [deletingStudentId, setDeletingStudentId] = useState(null);
  const [stats, setStats] = useState({
    totalStudents: 0,
    presentToday: 0,
    totalRecords: 0,
  });

  /**
   * Fetch all dashboard data
   */
  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch data in parallel
      const [todayRes, allRes, studentsRes] = await Promise.all([
        attendanceAPI.getToday(),
        attendanceAPI.getAll({ limit: 100 }),
        studentAPI.getAll(),
      ]);

      setTodayAttendance(todayRes.data.attendance || []);
      setAllAttendance(allRes.data.attendance || []);
      setStudents(studentsRes.data.students || []);

      setStats({
        totalStudents: studentsRes.data.count || 0,
        presentToday: todayRes.data.count || 0,
        totalRecords: allRes.data.total || 0,
      });

      console.log('[DASHBOARD] Data loaded successfully');
    } catch (error) {
      console.error('[DASHBOARD] Error fetching data:', error.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDeleteStudent = async (student) => {
    if (!student?._id) return;

    const ok = window.confirm(`Delete ${student.name} (${student.rollNumber})?`);
    if (!ok) return;

    try {
      setDeletingStudentId(student._id);
      await studentAPI.delete(student._id);
      await fetchData();
    } catch (error) {
      const msg = error?.response?.data?.error || error.message || 'Failed to delete student.';
      // keep it simple: alert is fine for this small feature
      window.alert(msg);
    } finally {
      setDeletingStudentId(null);
    }
  };

  /**
   * Filter attendance by date
   */
  const handleDateFilter = async () => {
    if (!dateFilter) return;
    try {
      const res = await attendanceAPI.getAll({ date: dateFilter });
      setAllAttendance(res.data.attendance || []);
    } catch (error) {
      console.error('[DASHBOARD] Filter error:', error.message);
    }
  };

  /**
   * Format date for display
   */
  const formatDate = (dateStr) => {
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-container">
          <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">📊 Attendance Dashboard</h1>
        <p className="page-subtitle">
          Overview of attendance records and registered students
        </p>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.totalStudents}</div>
          <div className="stat-label">Registered Students</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--success)' }}>
            {stats.presentToday}
          </div>
          <div className="stat-label">Present Today</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--warning)' }}>
            {Math.max(0, stats.totalStudents - stats.presentToday)}
          </div>
          <div className="stat-label">Absent Today</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--gray-500)' }}>
            {stats.totalRecords}
          </div>
          <div className="stat-label">Total Records</div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
        {[
          { id: 'today', label: "📅 Today's Attendance" },
          { id: 'all', label: '📁 All Records' },
          { id: 'students', label: '👥 Students' },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`btn ${activeTab === tab.id ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
        <button className="btn btn-outline" onClick={fetchData} style={{ marginLeft: 'auto' }}>
          🔄 Refresh
        </button>
      </div>

      {/* Today's Attendance Tab */}
      {activeTab === 'today' && (
        <div className="card">
          <h2 className="card-title">
            📅 Today's Attendance — {formatDate(new Date().toISOString())}
          </h2>

          {todayAttendance.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📭</div>
              <p className="empty-state-text">No attendance records for today yet.</p>
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>Roll Number</th>
                    <th>Time</th>
                    <th>Status</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {todayAttendance.map((record, index) => (
                    <tr key={record._id}>
                      <td>{index + 1}</td>
                      <td><strong>{record.studentId?.name || 'N/A'}</strong></td>
                      <td>{record.studentId?.rollNumber || 'N/A'}</td>
                      <td>{record.time}</td>
                      <td>
                        <span className={`badge badge-${record.status === 'present' ? 'success' : 'warning'}`}>
                          {record.status}
                        </span>
                      </td>
                      <td>{record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* All Records Tab */}
      {activeTab === 'all' && (
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <h2 className="card-title" style={{ margin: 0 }}>📁 All Attendance Records</h2>
            <div style={{ display: 'flex', gap: '0.5rem', marginLeft: 'auto' }}>
              <input
                type="date"
                className="form-input"
                style={{ width: 'auto' }}
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
              />
              <button className="btn btn-primary" onClick={handleDateFilter}>
                Filter
              </button>
              <button
                className="btn btn-outline"
                onClick={() => {
                  setDateFilter('');
                  fetchData();
                }}
              >
                Clear
              </button>
            </div>
          </div>

          {allAttendance.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📭</div>
              <p className="empty-state-text">No attendance records found.</p>
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>Roll Number</th>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Status</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {allAttendance.map((record, index) => (
                    <tr key={record._id}>
                      <td>{index + 1}</td>
                      <td><strong>{record.studentId?.name || 'N/A'}</strong></td>
                      <td>{record.studentId?.rollNumber || 'N/A'}</td>
                      <td>{formatDate(record.date)}</td>
                      <td>{record.time}</td>
                      <td>
                        <span className={`badge badge-${record.status === 'present' ? 'success' : 'warning'}`}>
                          {record.status}
                        </span>
                      </td>
                      <td>{record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Students Tab */}
      {activeTab === 'students' && (
        <div className="card">
          <h2 className="card-title">👥 Registered Students</h2>

          {students.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">👤</div>
              <p className="empty-state-text">No students registered yet.</p>
            </div>
          ) : (
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>Roll Number</th>
                    <th>Registered On</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {students.map((student, index) => (
                    <tr key={student._id}>
                      <td>{index + 1}</td>
                      <td><strong>{student.name}</strong></td>
                      <td>{student.rollNumber}</td>
                      <td>{formatDate(student.createdAt)}</td>
                      <td>
                        <button
                          className="btn btn-outline"
                          style={{
                            borderColor: 'rgba(239,68,68,0.4)',
                            color: 'rgb(239,68,68)',
                            background: 'rgba(239,68,68,0.08)',
                            padding: '0.4rem 0.7rem',
                          }}
                          onClick={() => handleDeleteStudent(student)}
                          disabled={deletingStudentId === student._id}
                          title="Delete student"
                        >
                          {deletingStudentId === student._id ? 'Deleting…' : 'Delete'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
